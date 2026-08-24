#!/usr/bin/env python3
"""Run one blur-aware Learn2Splat contract on heterogeneous scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import FusedSSIMMap
from PIL import Image, ImageDraw
from torch import Tensor

from optgs.dataset.colmap.utils import Parser
from optgs.dataset.data_types import BatchedViews
from optgs.experimental.api import OptGS
from optgs.experimental.blur_aware import (
    BlurAwareObjective,
    BlurAwareObjectiveConfig,
    estimate_evssm_reliability,
)
from optgs.experimental.initializers_utils import knn, points_to_gaussians
from optgs.model.types import Gaussians
from optgs.scene_trainer.common.gaussian_adapter import build_covariance

try:
    from experiments.blur_aware_cross_dataset.protocols import resolve_frame_protocol
except ModuleNotFoundError:  # Direct ``python path/to/run_cross_dataset.py``.
    from protocols import resolve_frame_protocol


NEAR_PLANE = 0.01
FAR_PLANE = 100.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-config", default=str(here / "scenes.json"))
    parser.add_argument(
        "--checkpoint",
        default=(
            "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
            "checkpoints/epoch_5-step_50000.ckpt"
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--initial-ply",
        default=None,
        help=(
            "Optional rollback-safe continuation initializer. Loads an existing "
            "3DGS PLY instead of rebuilding geometry from SfM/depth; intended "
            "for post-training convergence and capacity-schedule diagnostics."
        ),
    )
    parser.add_argument(
        "--initial-objective-state",
        default=None,
        help=(
            "Optional blur-aware objective checkpoint paired with --initial-ply. "
            "Restores the learned BPN/kernel state for cross-domain residual "
            "refinement; omitted for exact legacy behavior."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--decoder-backend",
        choices=("checkpoint", "fastgs"),
        default="checkpoint",
        help=(
            "Renderer used by both optimization and evaluation. LeGS requires "
            "fastgs because its exact per-Gaussian sensitivity is implemented "
            "inside the official LeGS FastGS rasterizer."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Update steps. Zero keeps the checkpoint's expected view exposure "
            "constant as scene size changes; positive values are smoke overrides."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        default="25%,50%,75%,100%",
        help="Comma-separated absolute steps or percentages of the run.",
    )
    parser.add_argument(
        "--num-init-points",
        type=int,
        default=0,
        help=(
            "Maximum number of observed geometry points used at initialization. "
            "Zero derives the budget from the released Learn2Splat checkpoint; "
            "a positive value provides an explicit reproducibility override."
        ),
    )
    parser.add_argument(
        "--max-sfm-points",
        type=int,
        default=-1,
        help=(
            "Optional cap on trusted geometry points. A non-positive value "
            "uses every available point up to --num-init-points."
        ),
    )
    parser.add_argument("--opt-batch-size", type=int, default=8)
    parser.add_argument(
        "--opt-batch-strategy",
        choices=(
            "checkpoint",
            "random",
            "sequential",
            "fps",
            "supervision_fps",
        ),
        default="supervision_fps",
        help=(
            "Per-step view sampler. supervision_fps realizes sharp w10 as a "
            "scene-level exposure distribution and preserves camera coverage "
            "without reading dataset labels."
        ),
    )
    parser.add_argument("--probe-views", type=int, default=8)
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=-1,
        help=(
            "Learned-state rollout length. -1 uses the checkpoint's training "
            "horizon, 0 disables restarts, and a positive value is an ablation."
        ),
    )
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS for short engineering gates; final runs compute it.",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--objective", choices=("blur-aware", "photometric"), default="blur-aware"
    )
    parser.add_argument(
        "--laplacian-loss-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of confidence-gated spatial Laplacian matching on non-sharp "
            "views. The validated cross-domain default is 0.1; use 0 for exact "
            "reconstruction-only rollback."
        ),
    )
    parser.add_argument(
        "--laplacian-loss-mode",
        choices=("spatial", "energy", "surplus"),
        default="spatial",
        help=(
            "spatial matches signed multiscale edges; surplus treats EVSSM as "
            "a one-sided edge floor and adapts its confidence from stable "
            "render-over-teacher gain; energy reproduces the old ablation."
        ),
    )
    parser.add_argument(
        "--coupled-dual-bpn",
        action="store_true",
        help=(
            "Use a shared blur-mode kernel with ordered EVSSM/RAW strengths. "
            "Without this flag the original single-RAW-kernel objective is exact."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("learned", "learned_projected", "adam"),
        default="learned",
        help=(
            "Update rule. learned_projected uses Learn2Splat for its released "
            "training horizon, then an objective-consistent Adam residual "
            "projection. Adam alone is a diagnostic."
        ),
    )
    parser.add_argument(
        "--adc",
        default="adaptive_legacy",
        help=(
            "OptGS ADC strategy. The validated cross-dataset default is "
            "adaptive_legacy; legs is the exact LeGS transplant and legs_blur "
            "adds blur-conditioned state and delayed reward."
        ),
    )
    parser.add_argument(
        "--densification-reward",
        choices=("off", "surplus_probe", "probe_control"),
        default="off",
        help=(
            "Delayed fixed-training-probe reward for adaptive densification. "
            "surplus_probe combines confidence-weighted EVSSM-target PSNR and "
            "supported render-over-EVSSM Laplacian surplus; off exactly "
            "restores the original ADC path; probe_control performs identical "
            "probe renders without feeding them to ADC for a fair ablation."
        ),
    )
    parser.add_argument(
        "--legs-blur-quality-weight",
        type=float,
        default=None,
        help="Override the legs_blur multi-view quality reward weight.",
    )
    parser.add_argument(
        "--legs-blur-capacity-weight",
        type=float,
        default=None,
        help="Override the legs_blur relative primitive-growth cost.",
    )
    parser.add_argument(
        "--legs-blur-start-iter",
        type=int,
        default=None,
        help="Override the first iteration of blur-policy conditioning.",
    )
    parser.add_argument(
        "--legs-blur-ramp-iters",
        type=int,
        default=None,
        help="Override the blur-policy conditioning ramp duration.",
    )
    parser.add_argument(
        "--legs-local-objective",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the complete blur-aware objective for LeGS's per-Gaussian "
            "local gradient state. --no-legs-local-objective is the exact "
            "target-only local-state ablation."
        ),
    )
    return parser.parse_args()


def configure_legs_blur_ablation(refiner_cfg, args: argparse.Namespace) -> None:
    """Apply explicit ablation overrides without changing exact LeGS."""
    if args.adc != "legs_blur":
        overrides = (
            args.legs_blur_quality_weight,
            args.legs_blur_capacity_weight,
            args.legs_blur_start_iter,
            args.legs_blur_ramp_iters,
            args.legs_local_objective,
        )
        if any(value is not None for value in overrides):
            raise ValueError("--legs-blur-* overrides require --adc legs_blur")
        return
    if args.legs_blur_quality_weight is not None:
        if args.legs_blur_quality_weight < 0:
            raise ValueError("legs_blur quality weight must be non-negative")
        refiner_cfg.blur_quality_weight = args.legs_blur_quality_weight
    if args.legs_blur_capacity_weight is not None:
        if args.legs_blur_capacity_weight < 0:
            raise ValueError("legs_blur capacity weight must be non-negative")
        refiner_cfg.blur_capacity_weight = args.legs_blur_capacity_weight
    if args.legs_blur_start_iter is not None:
        if args.legs_blur_start_iter < 0:
            raise ValueError("legs_blur start iteration must be non-negative")
        refiner_cfg.blur_condition_start_iter = args.legs_blur_start_iter
    if args.legs_blur_ramp_iters is not None:
        if args.legs_blur_ramp_iters <= 0:
            raise ValueError("legs_blur ramp duration must be positive")
        refiner_cfg.blur_condition_ramp_iters = args.legs_blur_ramp_iters
    if args.legs_local_objective is not None:
        refiner_cfg.local_objective_conditioned = args.legs_local_objective


def read_hold(data_dir: Path) -> int:
    markers = sorted(data_dir.glob("hold=*"))
    if len(markers) != 1:
        raise RuntimeError(f"expected exactly one hold=N marker in {data_dir}")
    return int(markers[0].name.split("=", 1)[1])


def image_files(directory: Path) -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG"}
    return sorted(path for path in directory.iterdir() if path.suffix in allowed)


class ImageResolver:
    def __init__(self, directory: str, mode: str = "stem"):
        self.directory = Path(directory)
        self.mode = mode
        self.files = image_files(self.directory)
        self.by_stem = {path.stem: path for path in self.files}
        if not self.files:
            raise RuntimeError(f"no images found in {self.directory}")

    def resolve(self, image_name: str) -> Path:
        stem = Path(image_name).stem
        if self.mode == "sorted_index":
            index = int(stem)
            if index >= len(self.files):
                raise IndexError(f"raw index {index} outside {len(self.files)} files")
            return self.files[index]
        for candidate in (stem, stem.zfill(3), stem.zfill(6)):
            if candidate in self.by_stem:
                return self.by_stem[candidate]
        raise FileNotFoundError(f"no image matching {image_name!r} in {self.directory}")


def resize_depth_preserve_samples(
    depth: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Resize depth without discarding isolated measurements on downsampling."""
    source_height, source_width = depth.shape[:2]
    output_width, output_height = output_size
    if (source_width, source_height) == output_size:
        return np.ascontiguousarray(depth)
    if output_width > source_width or output_height > source_height:
        return cv2.resize(depth, output_size, interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(depth) & (depth > 0)
    # When measurements outnumber output pixels, collisions are unavoidable
    # and depth is an oversampled field: use conventional nearest resampling.
    # Otherwise every isolated measurement can in principle be retained, so
    # forward-project it instead of asking an inverse sampler to hit it.
    if int(valid.sum()) > output_width * output_height:
        return cv2.resize(depth, output_size, interpolation=cv2.INTER_NEAREST)
    resized = np.zeros((output_height, output_width), dtype=depth.dtype)
    if not valid.any():
        return resized

    source_y, source_x = np.nonzero(valid)
    target_x = np.floor((source_x + 0.5) * output_width / source_width).astype(
        np.int64
    )
    target_y = np.floor((source_y + 0.5) * output_height / source_height).astype(
        np.int64
    )
    np.clip(target_x, 0, output_width - 1, out=target_x)
    np.clip(target_y, 0, output_height - 1, out=target_y)

    target_flat = target_y * output_width + target_x
    nearest = np.full(output_height * output_width, np.inf, dtype=np.float64)
    np.minimum.at(nearest, target_flat, depth[valid].astype(np.float64))
    populated = np.isfinite(nearest)
    resized.reshape(-1)[populated] = nearest[populated].astype(depth.dtype)
    return resized


@dataclass(frozen=True)
class CameraPreprocessor:
    """Calibration-preserving image transform shared by every dataset."""

    source_size: tuple[int, int]
    pre_crop: tuple[int, int, int, int]
    resize_size: tuple[int, int]
    crop: tuple[int, int, int, int]
    distortion: tuple[float, ...]
    undistort_depth: bool

    @property
    def cropped_source_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.pre_crop
        return (
            self.source_size[0] - left - right,
            self.source_size[1] - top - bottom,
        )

    @property
    def output_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.crop
        return (
            self.resize_size[0] - left - right,
            self.resize_size[1] - top - bottom,
        )

    def transform_intrinsics(self, intrinsics: Tensor) -> Tensor:
        transformed = intrinsics.clone()
        transformed[0, 2] -= self.pre_crop[0]
        transformed[1, 2] -= self.pre_crop[1]
        scale_x = self.resize_size[0] / self.cropped_source_size[0]
        scale_y = self.resize_size[1] / self.cropped_source_size[1]
        transformed[0, :] *= scale_x
        transformed[1, :] *= scale_y
        transformed[0, 2] -= self.crop[0]
        transformed[1, 2] -= self.crop[1]
        return transformed

    def transform_array(
        self,
        array: np.ndarray,
        intrinsics: Tensor,
        *,
        is_depth: bool,
    ) -> np.ndarray:
        if (array.shape[1], array.shape[0]) != self.source_size:
            if is_depth:
                array = resize_depth_preserve_samples(array, self.source_size)
            else:
                array = cv2.resize(
                    array, self.source_size, interpolation=cv2.INTER_LANCZOS4
                )
        if self.distortion and (not is_depth or self.undistort_depth):
            interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            camera_matrix = intrinsics.detach().cpu().numpy().astype(np.float64)
            map_x, map_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                np.asarray(self.distortion, dtype=np.float64),
                None,
                camera_matrix,
                self.source_size,
                cv2.CV_32FC1,
            )
            array = cv2.remap(
                array,
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
            )
        pre_left, pre_top, pre_right, pre_bottom = self.pre_crop
        pre_y_stop = array.shape[0] - pre_bottom if pre_bottom else array.shape[0]
        pre_x_stop = array.shape[1] - pre_right if pre_right else array.shape[1]
        array = array[pre_top:pre_y_stop, pre_left:pre_x_stop]
        if self.resize_size != self.cropped_source_size:
            if is_depth:
                array = resize_depth_preserve_samples(array, self.resize_size)
            else:
                array = cv2.resize(
                    array, self.resize_size, interpolation=cv2.INTER_LANCZOS4
                )
        left, top, right, bottom = self.crop
        y_stop = array.shape[0] - bottom if bottom else array.shape[0]
        x_stop = array.shape[1] - right if right else array.shape[1]
        return np.ascontiguousarray(array[top:y_stop, left:x_stop])


def build_camera_preprocessors(parser: Parser, cfg: dict) -> list[CameraPreprocessor]:
    """Build one calibrated transform per view, with a shared output shape.

    COLMAP scenes can legitimately mix cameras and image sizes. Per-camera
    metadata handles that geometry without a dataset-name branch; the learned
    optimizer still receives one homogeneous image tensor.
    """
    common_spec = cfg.get("camera_preprocess", {})
    camera_specs = cfg.get("camera_preprocess_by_camera_id", {})
    preprocessors = []
    for camera_id in parser.camera_ids:
        source_size = tuple(int(value) for value in parser.imsize_dict[camera_id])
        spec = {**common_spec, **camera_specs.get(str(camera_id), {})}
        pre_crop = tuple(int(value) for value in spec.get("pre_crop", (0, 0, 0, 0)))
        if len(pre_crop) != 4:
            raise ValueError("camera_preprocess pre_crop must have 4 entries")
        pre_size = (
            source_size[0] - pre_crop[0] - pre_crop[2],
            source_size[1] - pre_crop[1] - pre_crop[3],
        )
        resize_size = tuple(int(value) for value in spec.get("resize", pre_size))
        crop = tuple(int(value) for value in spec.get("crop", (0, 0, 0, 0)))
        if len(resize_size) != 2 or len(crop) != 4:
            raise ValueError("camera_preprocess resize/crop must have 2/4 entries")
        preprocessor = CameraPreprocessor(
            source_size=source_size,
            pre_crop=pre_crop,
            resize_size=resize_size,
            crop=crop,
            distortion=tuple(float(value) for value in spec.get("distortion", ())),
            undistort_depth=bool(spec.get("undistort_depth", True)),
        )
        if min(preprocessor.cropped_source_size + preprocessor.output_size) <= 0:
            raise ValueError("camera_preprocess crop removes the complete image")
        preprocessors.append(preprocessor)
    output_sizes = {preprocessor.output_size for preprocessor in preprocessors}
    if len(output_sizes) != 1:
        raise ValueError(f"all camera preprocessors must share one output size: {output_sizes}")
    return preprocessors


def load_rgb(
    path: Path, intrinsics: Tensor, preprocessor: CameraPreprocessor
) -> Tensor:
    array = np.asarray(Image.open(path).convert("RGB"))
    array = preprocessor.transform_array(array, intrinsics, is_depth=False)
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_depth(
    path: Path, intrinsics: Tensor, preprocessor: CameraPreprocessor
) -> Tensor:
    array = np.asarray(Image.open(path), dtype=np.float32)
    array = preprocessor.transform_array(array, intrinsics, is_depth=True)
    return torch.from_numpy(array.copy()) / 1000.0


def depth_measurement_to_z(
    measurement: Tensor,
    x: Tensor,
    y: Tensor,
    intrinsics: Tensor,
    convention: str,
) -> Tensor:
    """Convert a calibrated depth measurement to camera-axis depth."""
    if convention == "z":
        return measurement
    if convention == "range":
        normalized_x = (x.float() - intrinsics[0, 2]) / intrinsics[0, 0]
        normalized_y = (y.float() - intrinsics[1, 2]) / intrinsics[1, 1]
        ray_norm = torch.sqrt(1.0 + normalized_x.square() + normalized_y.square())
        return measurement / ray_norm
    raise ValueError(f"unsupported depth convention {convention!r}")


def build_valid_mask(
    intrinsics: Tensor, preprocessor: CameraPreprocessor
) -> Tensor:
    """Return the valid sensor domain after calibration-preserving warping."""
    source = np.ones(
        (preprocessor.source_size[1], preprocessor.source_size[0]), dtype=np.float32
    )
    transformed = preprocessor.transform_array(source, intrinsics, is_depth=False)
    return torch.from_numpy(transformed > (1.0 - 1e-6)).unsqueeze(0)


def authoritative_all_sharp(data_dir: Path, num_views: int) -> bool:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (
        manifest.get("official_sharp_present") == num_views
        and manifest.get("official_sharp_missing_raw") == []
    )


def resolve_evaluation_indices(
    parser: Parser, cfg: dict, data_dir: Path, hold: int
) -> tuple[list[int], str]:
    """Resolve an explicit benchmark split before considering hold=N.

    Generated training views must never enter evaluation by arithmetic
    accident. A manifest-bound name list is therefore authoritative whenever
    one is provided; hold=N remains the legacy Deblur-NeRF fallback.
    """
    manifest_spec = cfg.get("evaluation_manifest")
    if manifest_spec is not None:
        manifest = json.loads(Path(manifest_spec["path"]).read_text())
        manifest_key = manifest_spec.get("key")
        if manifest_key is None:
            if not isinstance(manifest, list):
                raise TypeError("keyless evaluation manifest must be a JSON list")
            manifest_names = manifest
            manifest_label = manifest_spec.get("label", "name_list")
        else:
            manifest_names = manifest[manifest_key]
            manifest_label = manifest_key
        names = {Path(name).stem for name in manifest_names}
        index_by_name = {
            Path(image_name).stem: index
            for index, image_name in enumerate(parser.image_names)
        }
        missing = names - index_by_name.keys()
        if missing:
            raise RuntimeError(f"evaluation images absent from COLMAP scene: {sorted(missing)}")
        train_only_path = cfg.get("train_only_manifest")
        if train_only_path is not None:
            train_only = {
                Path(name).stem
                for name in json.loads(Path(train_only_path).read_text())["image_names"]
            }
            overlap = names & train_only
            if overlap:
                raise RuntimeError(f"train-only images leaked into evaluation: {sorted(overlap)}")
        indices = sorted(index_by_name[name] for name in names)
        return indices, f"manifest:{manifest_label}"
    if authoritative_all_sharp(data_dir, len(parser.image_names)):
        return list(range(len(parser.image_names))), "authoritative_scene_manifest"
    return (
        [index for index in range(len(parser.image_names)) if index % hold == 0],
        f"hold={hold}",
    )


def resolve_sharp_supervision(
    parser: Parser, cfg: dict, evaluation_indices: list[int]
) -> tuple[set[str], list[str], str]:
    """Resolve training-time sharp labels independently of evaluation views.

    Some benchmarks provide a sharp reference for every evaluated camera. That
    describes metric availability, not whether the corresponding training
    observation should bypass EVSSM/BPN. The explicit policy prevents the two
    contracts from being coupled accidentally.
    """
    sharp_names = {
        Path(name).stem
        for name in json.loads(Path(cfg["sharp_json"]).read_text())
    }
    sharp_sources = ["sharp_json"]
    policy = cfg.get("sharp_supervision_policy", "evaluation_is_sharp")
    if policy == "sharp_json_only":
        return sharp_names, sharp_sources, policy
    if policy != "evaluation_is_sharp":
        raise ValueError(f"unsupported sharp supervision policy {policy!r}")

    if authoritative_all_sharp(Path(cfg["data_dir"]), len(parser.image_names)):
        sharp_names.update(Path(name).stem for name in parser.image_names)
        sharp_sources.append("authoritative_scene_manifest")
    sharp_names.update(
        Path(parser.image_names[index]).stem for index in evaluation_indices
    )
    sharp_sources.append("evaluation_indices")
    return sharp_names, sharp_sources, policy


def resolve_auxiliary_supervision(cfg: dict) -> tuple[dict[str, float], list[str]]:
    """Resolve confidence-weighted train-only pseudo observations.

    Auxiliary direct supervision is deliberately independent of known-sharp
    supervision: it never receives w10 and never changes an evaluation split.
    """
    manifest_path = cfg.get("auxiliary_supervision_manifest")
    if manifest_path is None:
        return {}, []
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest.get("views")
    if not isinstance(rows, list):
        raise TypeError("auxiliary supervision manifest must contain a views list")
    confidence_by_name: dict[str, float] = {}
    for row in rows:
        name = Path(row["image_name"]).stem
        confidence = float(row["confidence"])
        if name in confidence_by_name:
            raise RuntimeError(f"duplicate auxiliary supervision image: {name}")
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise ValueError(
                f"auxiliary confidence for {name} must be finite and in (0,1]"
            )
        confidence_by_name[name] = confidence
    return confidence_by_name, [f"manifest:{Path(manifest_path).name}"]


def _indices_for_names(parser: Parser, names: list[str], label: str) -> list[int]:
    index_by_name = {
        Path(image_name).stem: index
        for index, image_name in enumerate(parser.image_names)
    }
    normalized = [Path(name).stem for name in names]
    missing = [name for name in normalized if name not in index_by_name]
    if missing:
        preview = missing[:20]
        raise RuntimeError(
            f"{label} images absent from COLMAP scene ({len(missing)} total): {preview}"
        )
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"{label} image list contains duplicates")
    return [index_by_name[name] for name in normalized]


def resolve_scene_indices(
    parser: Parser, cfg: dict, data_dir: Path, hold: int
) -> tuple[list[int], list[int], str, dict[str, object] | None]:
    """Resolve immutable input/evaluation protocol without optimizer heuristics."""
    protocol_spec = cfg.get("frame_protocol")
    if protocol_spec is None:
        evaluation, source = resolve_evaluation_indices(parser, cfg, data_dir, hold)
        return list(range(len(parser.image_names))), evaluation, source, None

    resolved = resolve_frame_protocol(protocol_spec)
    optimization = _indices_for_names(
        parser, resolved.optimization_names, "optimization protocol"
    )
    evaluation = _indices_for_names(
        parser, resolved.evaluation_names, "evaluation protocol"
    )
    if not set(evaluation).issubset(optimization):
        raise RuntimeError("evaluation protocol is not a subset of the input stream")
    return optimization, evaluation, str(resolved.metadata["type"]), resolved.metadata


def resolve_initialization_budget(requested: int, optgs: OptGS) -> tuple[int, str]:
    """Resolve one architecture-conditioned budget without dataset heuristics."""
    if requested < 0:
        raise ValueError("--num-init-points must be zero or positive")
    if requested > 0:
        return requested, "explicit_override"
    budget = int(getattr(optgs, "reference_initial_gaussians", 0))
    if budget <= 0:
        raise RuntimeError(
            "checkpoint does not declare a positive reference initialization; "
            "pass --num-init-points explicitly"
        )
    return budget, "checkpoint_reference_initialization"


def resolve_update_budget(
    requested: int,
    checkpoint_steps: int,
    num_views: int,
    reference_views: int,
    supervision_mass: Tensor | None = None,
) -> tuple[int, str]:
    """Keep expected optimizer exposure per unit supervision risk constant."""
    if requested < 0:
        raise ValueError("--steps must be zero or positive")
    if requested > 0:
        return requested, "explicit_override"
    if min(checkpoint_steps, num_views, reference_views) <= 0:
        raise ValueError("automatic update-budget inputs must be positive")
    effective_views = float(num_views)
    source = "checkpoint_view_exposure"
    if supervision_mass is not None:
        mass = supervision_mass.float()
        if (
            mass.numel() != num_views
            or not bool(torch.isfinite(mass).all())
            or bool((mass <= 0).any())
        ):
            raise ValueError("supervision_mass must contain one positive finite value per view")
        # The sampler visits a maximum-mass (sharp) observation at the same
        # expected frequency that the released checkpoint visited one of its
        # reference context views. Lower-mass RAW/BPN observations receive the
        # intended relative exposure rather than forcing a dataset-size-scaled
        # recurrent horizon.
        effective_views = float(mass.sum() / mass.max())
        source = "checkpoint_supervision_risk_exposure"
    steps = max(
        checkpoint_steps,
        math.ceil(checkpoint_steps * effective_views / reference_views),
    )
    return steps, source


def resolve_eval_steps(spec: str, num_steps: int) -> list[int]:
    steps = {num_steps}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("%"):
            value = float(token[:-1]) / 100.0
            if not 0.0 < value <= 1.0:
                raise ValueError(f"evaluation percentage outside (0,100]: {token}")
            step = max(1, round(value * num_steps))
        else:
            step = int(token)
        if 0 < step <= num_steps:
            steps.add(step)
    return sorted(steps)


def collect_scene(parser: Parser, cfg: dict, evaluation_indices: list[int]) -> dict:
    raw_resolver = ImageResolver(cfg["raw_dir"], cfg.get("raw_mode", "stem"))
    raw_fallback_resolver = (
        ImageResolver(cfg["raw_fallback_dir"], "stem")
        if cfg.get("raw_fallback_dir")
        else None
    )
    evssm_resolver = ImageResolver(cfg["evssm_dir"], "stem")
    sharp_names, sharp_sources, sharp_policy = resolve_sharp_supervision(
        parser, cfg, evaluation_indices
    )
    auxiliary_confidence, auxiliary_sources = resolve_auxiliary_supervision(cfg)
    parser_names = {Path(name).stem for name in parser.image_names}
    missing_auxiliary = auxiliary_confidence.keys() - parser_names
    if missing_auxiliary:
        raise RuntimeError(
            "auxiliary supervision images absent from COLMAP scene: "
            f"{sorted(missing_auxiliary)}"
        )
    sharp_auxiliary_overlap = sharp_names & auxiliary_confidence.keys()
    if sharp_auxiliary_overlap:
        raise RuntimeError(
            "auxiliary observations cannot also be authoritative sharp: "
            f"{sorted(sharp_auxiliary_overlap)}"
        )
    train_only_path = cfg.get("train_only_manifest")
    if auxiliary_confidence and train_only_path is None:
        raise RuntimeError("auxiliary supervision requires a train-only manifest")
    if train_only_path is not None:
        train_only_names = {
            Path(name).stem
            for name in json.loads(Path(train_only_path).read_text())["image_names"]
        }
        outside_train_only = auxiliary_confidence.keys() - train_only_names
        if outside_train_only:
            raise RuntimeError(
                "auxiliary supervision is not declared train-only: "
                f"{sorted(outside_train_only)}"
            )
    camera_preprocessors = build_camera_preprocessors(parser, cfg)
    intrinsics = [
        torch.from_numpy(parser.Ks_dict[camera_id]).float()
        for camera_id in parser.camera_ids
    ]

    raw_images, target_images, known_sharp, valid_masks = [], [], [], []
    direct_supervision, supervision_confidence, auxiliary_mask = [], [], []
    raw_paths, target_paths = [], []
    for index, image_name in enumerate(parser.image_names):
        stem = Path(image_name).stem
        try:
            raw_path = raw_resolver.resolve(image_name)
        except (FileNotFoundError, IndexError, ValueError):
            if raw_fallback_resolver is None:
                raise
            raw_path = raw_fallback_resolver.resolve(image_name)
        evssm_path = evssm_resolver.resolve(image_name)
        is_sharp = stem in sharp_names
        is_auxiliary = stem in auxiliary_confidence
        is_direct = is_sharp or is_auxiliary
        preprocessor = camera_preprocessors[index]
        raw = load_rgb(raw_path, intrinsics[index], preprocessor)
        target = (
            raw.clone()
            if is_direct
            else load_rgb(evssm_path, intrinsics[index], preprocessor)
        )
        raw_images.append(raw)
        target_images.append(target)
        known_sharp.append(is_sharp)
        direct_supervision.append(is_direct)
        supervision_confidence.append(auxiliary_confidence.get(stem, 1.0))
        auxiliary_mask.append(is_auxiliary)
        valid_masks.append(build_valid_mask(intrinsics[index], preprocessor))
        raw_paths.append(str(raw_path))
        target_paths.append(str(raw_path if is_direct else evssm_path))

    raw_images = torch.stack(raw_images)
    target_images = torch.stack(target_images)
    known_sharp_tensor = torch.tensor(known_sharp, dtype=torch.bool)
    direct_supervision_tensor = torch.tensor(direct_supervision, dtype=torch.bool)
    supervision_confidence_tensor = torch.tensor(
        supervision_confidence, dtype=torch.float32
    )
    sampling_mass = torch.where(
        known_sharp_tensor,
        torch.full_like(known_sharp_tensor, 10.0, dtype=torch.float32),
        torch.ones_like(known_sharp_tensor, dtype=torch.float32),
    )
    confidence_chunks = []
    reliability_chunks: dict[str, list[Tensor]] = {}
    # Full TUM streams contain thousands of frames. Reliability is an
    # independent per-frame statistic, so bounded chunks are exactly
    # equivalent and avoid a dataset-size-dependent memory spike.
    for start in range(0, len(raw_images), 32):
        stop = min(start + 32, len(raw_images))
        confidence_chunk, diagnostic_chunk = estimate_evssm_reliability(
            raw_images[start:stop],
            target_images[start:stop],
            known_sharp_tensor[start:stop],
        )
        confidence_chunks.append(confidence_chunk)
        for key, value in diagnostic_chunk.items():
            reliability_chunks.setdefault(key, []).append(value)
    confidence = torch.cat(confidence_chunks)
    reliability = {
        key: torch.cat(chunks) for key, chunks in reliability_chunks.items()
    }
    return {
        "raw_images": raw_images,
        "target_images": target_images,
        "known_sharp": known_sharp_tensor,
        "direct_supervision": direct_supervision_tensor,
        "supervision_confidence": supervision_confidence_tensor,
        "auxiliary_mask": torch.tensor(auxiliary_mask, dtype=torch.bool),
        "sampling_mass": sampling_mass,
        "valid_mask": torch.stack(valid_masks),
        "confidence": confidence,
        "reliability": reliability,
        "c2w": torch.from_numpy(parser.camtoworlds).float(),
        "intrinsics": torch.stack(
            [
                preprocessor.transform_intrinsics(value)
                for preprocessor, value in zip(camera_preprocessors, intrinsics)
            ]
        ),
        "raw_paths": raw_paths,
        "target_paths": target_paths,
        "size": camera_preprocessors[0].output_size,
        "camera_preprocessors": camera_preprocessors,
        "sharp_sources": sharp_sources,
        "sharp_policy": sharp_policy,
        "auxiliary_sources": auxiliary_sources,
    }


def build_views(
    scene: dict,
    indices: list[int],
    scene_scale: float,
    device: torch.device,
    *,
    policy_probe_indices: set[int] | None = None,
) -> BatchedViews:
    selection = torch.tensor(indices, dtype=torch.long)
    images = scene["target_images"][selection]
    raw = scene["raw_images"][selection]
    c2w = scene["c2w"][selection]
    intrinsics = scene["intrinsics"][selection].clone()
    h, w = images.shape[-2:]
    intrinsics[:, 0, :] /= w
    intrinsics[:, 1, :] /= h

    def batch(tensor: Tensor, dtype: torch.dtype | None = torch.float32) -> Tensor:
        tensor = tensor.unsqueeze(0).to(device=device)
        return tensor if dtype is None else tensor.to(dtype=dtype)

    v = len(indices)
    policy_probe = torch.tensor(
        [
            policy_probe_indices is not None and index in policy_probe_indices
            for index in indices
        ],
        dtype=torch.bool,
    )
    return BatchedViews.from_dict(
        {
            "extrinsics": batch(c2w),
            "intrinsics": batch(intrinsics),
            "image": batch(images),
            "raw_image": batch(raw),
            "target_confidence": batch(scene["confidence"][selection]),
            "known_sharp": batch(scene["known_sharp"][selection], dtype=None),
            "direct_supervision": batch(
                scene["direct_supervision"][selection], dtype=None
            ),
            "supervision_confidence": batch(
                scene["supervision_confidence"][selection]
            ),
            "sampling_mass": batch(scene["sampling_mass"][selection]),
            "valid_mask": batch(scene["valid_mask"][selection], dtype=None),
            "policy_probe": batch(policy_probe, dtype=None),
            "near": torch.full((1, v), NEAR_PLANE, device=device),
            "far": torch.full((1, v), FAR_PLANE, device=device),
            "index": torch.arange(v, device=device).unsqueeze(0),
            "scene_scale": torch.tensor([scene_scale], device=device),
        }
    )


def resolve_depth_samples_per_view(
    needed: int, available_depth_views: int
) -> int:
    if needed <= 0 or available_depth_views <= 0:
        return 0
    return max(1, math.ceil(needed / available_depth_views))


def depth_fused_initialization(
    parser: Parser,
    scene: dict,
    cfg: dict,
    train_indices: list[int],
    *,
    target_count: int,
    max_sfm_points: int,
    sh_degree: int,
    device: torch.device,
    seed: int,
) -> tuple[Gaussians, dict]:
    generator = torch.Generator().manual_seed(seed)
    errors = torch.from_numpy(parser.points_err).float()
    sfm_limit = target_count if max_sfm_points <= 0 else max_sfm_points
    # Match Learn2Splat's SfM-initialized training distribution whenever the
    # scene provides enough trusted geometry. Depth only fills the remaining
    # budget; there is no dataset-specific SfM/depth mixing ratio.
    sfm_count = min(sfm_limit, target_count, parser.points.shape[0])
    sfm_order = torch.argsort(errors)[:sfm_count]
    sfm_points = torch.from_numpy(parser.points).float()[sfm_order]
    sfm_colors = torch.from_numpy(parser.points_rgb.astype(np.float32) / 255.0)[sfm_order]

    needed = target_count - sfm_count
    sampled_points, sampled_colors, sampled_depths = [], [], []
    missing_depth_views = 0
    depth_quantile = None
    if needed > 0:
        depth_dir = Path(cfg["depth_dir"])
        depth_records = []
        for index in train_indices:
            stem = Path(parser.image_names[index]).stem
            depth_path = depth_dir / f"{stem}.png"
            if not depth_path.exists():
                # Virtual or auxiliary cameras may provide calibrated RGB but
                # no sensor depth. SfM still constrains those views; depth
                # fusion simply uses the observations that actually exist.
                missing_depth_views += 1
                continue
            depth = load_depth(
                depth_path,
                torch.from_numpy(parser.Ks_dict[parser.camera_ids[index]]).float(),
                scene["camera_preprocessors"][index],
            )
            valid = torch.nonzero(
                torch.isfinite(depth) & (depth > 0.0), as_tuple=False
            )
            if valid.numel() == 0:
                continue
            depth_records.append((index, depth, valid))

        # Auxiliary RGB cameras often have no sensor depth. Allocate the depth
        # budget only across observations that can contribute geometry so that
        # adding such cameras cannot dilute an otherwise identical initializer.
        per_view = resolve_depth_samples_per_view(needed, len(depth_records))
        for index, depth, valid in depth_records:
            if valid.shape[0] > per_view:
                stride = valid.shape[0] / per_view
                positions = (torch.arange(per_view) * stride).long()
                offset = int(
                    torch.randint(max(1, int(stride)), (1,), generator=generator)
                )
                positions = (positions + offset).clamp_max(valid.shape[0] - 1)
                valid = valid[positions]
            y, x = valid[:, 0], valid[:, 1]
            K = scene["intrinsics"][index]
            z = depth_measurement_to_z(
                depth[y, x],
                x,
                y,
                K,
                cfg.get("depth_convention", "z"),
            )
            camera_points = torch.stack(
                (
                    (x.float() - K[0, 2]) * z / K[0, 0],
                    (y.float() - K[1, 2]) * z / K[1, 1],
                    z,
                ),
                dim=1,
            )
            c2w = scene["c2w"][index]
            sampled_points.append(camera_points @ c2w[:3, :3].T + c2w[:3, 3])
            sampled_colors.append(scene["target_images"][index, :, y, x].T)
            sampled_depths.append(z)

        if not sampled_points and sfm_count == 0:
            raise RuntimeError("no valid geometry samples for initialization")
        if sampled_points:
            depth_points = torch.cat(sampled_points)
            depth_colors = torch.cat(sampled_colors)
            depth_values = torch.cat(sampled_depths)
            lo, hi = torch.quantile(depth_values, torch.tensor([0.005, 0.995]))
            valid = (depth_values >= lo) & (depth_values <= hi)
            depth_points, depth_colors = depth_points[valid], depth_colors[valid]
            # target_count is an upper bound. Repeating sparse geometry to hit
            # it creates exact duplicate Gaussians and a domain-dependent scale
            # distribution, so retain only observed geometry.
            available = min(needed, depth_points.shape[0])
            depth_points, depth_colors = (
                depth_points[:available],
                depth_colors[:available],
            )
            depth_quantile = [float(lo), float(hi)]
        else:
            depth_points = torch.empty((0, 3), dtype=sfm_points.dtype)
            depth_colors = torch.empty((0, 3), dtype=sfm_colors.dtype)
    else:
        depth_points = torch.empty((0, 3), dtype=sfm_points.dtype)
        depth_colors = torch.empty((0, 3), dtype=sfm_colors.dtype)
    points = torch.cat((sfm_points, depth_points), dim=0)
    colors = torch.cat((sfm_colors, depth_colors), dim=0).clamp(0.0, 1.0)

    distances = knn(points, 4)[:, 1:].square().mean(dim=1).sqrt().clamp_min(1e-6)
    scales = distances[:, None].repeat(1, 3)
    opacities = torch.full((points.shape[0],), 0.1)
    gaussian_dict = points_to_gaussians(
        {"xyz": points, "rgb": colors, "scales": scales, "opacities": opacities},
        sh_degree=sh_degree,
        device=device,
    )
    harmonics = torch.cat(
        (gaussian_dict["sh0"], gaussian_dict["shN"]), dim=1
    ).permute(0, 2, 1)
    scales_active = torch.exp(gaussian_dict["scales_raw"])
    opacities_active = torch.sigmoid(gaussian_dict["opacities_raw"])
    rotations = F.normalize(gaussian_dict["rotations_unnorm"], dim=-1)
    covariances = build_covariance(scale=scales_active, rotation_xyzw=rotations)

    def batch(value: Tensor) -> Tensor:
        return value.unsqueeze(0).float()

    return Gaussians(
        means=batch(gaussian_dict["xyz"]),
        covariances=batch(covariances),
        harmonics=batch(harmonics),
        opacities=batch(opacities_active),
        scales=batch(scales_active),
        rotations=batch(rotations),
        rotations_unnorm=batch(gaussian_dict["rotations_unnorm"]),
    ), {
        "requested_max": int(target_count),
        "total": int(points.shape[0]),
        "sfm": int(sfm_count),
        "depth": int(depth_points.shape[0]),
        "unique": int(torch.unique(points, dim=0).shape[0]),
        "depth_quantile_m": depth_quantile,
        "missing_depth_views": int(missing_depth_views),
        "depth_convention": cfg.get("depth_convention", "z"),
    }


def farthest_probe_indices(c2w: Tensor, count: int) -> list[int]:
    positions = c2w[:, :3, 3]
    count = min(count, positions.shape[0])
    selected = [0]
    min_distance = torch.full((positions.shape[0],), float("inf"))
    for _ in range(1, count):
        distance = torch.linalg.norm(positions - positions[selected[-1]], dim=1)
        min_distance = torch.minimum(min_distance, distance)
        selected.append(int(torch.argmax(min_distance)))
    return selected


@torch.no_grad()
def render_metrics(
    optgs: OptGS,
    gaussians: Gaussians,
    views: BatchedViews,
    *,
    lpips_model: torch.nn.Module | None,
    metric_batch_size: int,
) -> dict:
    h, w = views.image.shape[-2:]
    output = optgs.decoder.forward(
        gaussians,
        views.extrinsics,
        views.intrinsics,
        views.near,
        views.far,
        image_shape=(h, w),
    )
    prediction = output.color.clamp(0.0, 1.0)
    target = views.raw_image if views.raw_image is not None else views.image
    squared_error = (prediction - target).square()
    if views.valid_mask is None:
        mse = squared_error.mean(dim=(2, 3, 4))
        valid_fraction = torch.ones_like(mse)
    else:
        valid = views.valid_mask.to(dtype=squared_error.dtype)
        denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0) * target.shape[2]
        mse = (squared_error * valid).sum(dim=(2, 3, 4)) / denominator
        valid_fraction = valid.mean(dim=(2, 3, 4))
    mse = mse.clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    flat_prediction = prediction.flatten(0, 1).contiguous()
    flat_target = target.flatten(0, 1).contiguous()
    ssim_map = FusedSSIMMap.apply(
        0.01**2, 0.03**2, flat_prediction, flat_target, "valid", False
    )
    ssim = ssim_map.mean(dim=(1, 2, 3))
    per_view_lpips = None
    if lpips_model is not None:
        if metric_batch_size <= 0:
            raise ValueError("metric_batch_size must be positive")
        flat_valid = (
            None
            if views.valid_mask is None
            else views.valid_mask.flatten(0, 1).to(dtype=flat_prediction.dtype)
        )
        lpips_values = []
        for start in range(0, len(flat_prediction), metric_batch_size):
            stop = min(start + metric_batch_size, len(flat_prediction))
            pred_chunk = flat_prediction[start:stop]
            target_chunk = flat_target[start:stop]
            if flat_valid is not None:
                valid_chunk = flat_valid[start:stop]
                # Invalid calibration borders are identical in both arguments,
                # so they cannot be rewarded or penalized by LPIPS.
                pred_chunk = pred_chunk * valid_chunk + target_chunk * (
                    1.0 - valid_chunk
                )
            lpips_values.append(
                lpips_model(pred_chunk, target_chunk, normalize=True).flatten()
            )
        per_view_lpips = torch.cat(lpips_values)
    return {
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.mean()),
        "lpips": None if per_view_lpips is None else float(per_view_lpips.mean()),
        "per_view_psnr": [float(value) for value in psnr.flatten()],
        "per_view_lpips": (
            None
            if per_view_lpips is None
            else [float(value) for value in per_view_lpips]
        ),
        "valid_fraction": float(valid_fraction.mean()),
        "prediction": prediction[0].cpu(),
    }


@torch.no_grad()
def render_densification_probe(
    optgs: OptGS,
    gaussians: Gaussians,
    views: BatchedViews,
    objective: BlurAwareObjective,
) -> dict[str, float | bool]:
    """Evaluate ADC reward on a fixed subset of optimization views only."""
    h, w = views.image.shape[-2:]
    output = optgs.decoder.forward(
        gaussians,
        views.extrinsics,
        views.intrinsics,
        views.near,
        views.far,
        image_shape=(h, w),
    )
    prediction = output.color.clamp(0.0, 1.0)
    target = views.image
    squared_error = (prediction - target).square()
    if views.valid_mask is None:
        mse = squared_error.mean(dim=(2, 3, 4))
    else:
        valid = views.valid_mask.to(dtype=squared_error.dtype)
        denominator = valid.sum(dim=(2, 3, 4)).clamp_min(1.0) * target.shape[2]
        mse = (squared_error * valid).sum(dim=(2, 3, 4)) / denominator
    per_view_psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    confidence = (
        torch.ones_like(per_view_psnr)
        if views.target_confidence is None
        else views.target_confidence.clamp(0.0, 1.0)
    )
    confidence_sum = confidence.sum()
    if float(confidence_sum) == 0.0:
        confidence = torch.ones_like(confidence)
        confidence_sum = confidence.sum()
    probe_psnr = (per_view_psnr * confidence).sum() / confidence_sum
    known_sharp = (
        torch.zeros_like(per_view_psnr, dtype=torch.bool)
        if views.known_sharp is None
        else views.known_sharp.bool()
    )
    raw = views.raw_image if views.raw_image is not None else target
    surplus = objective.measure_probe_surplus(
        prediction,
        raw,
        target,
        known_sharp,
        confidence,
        views.valid_mask,
    )
    return {
        "probe_psnr": float(probe_psnr),
        "probe_surplus": float(surplus["surplus"]),
        "probe_teacher_gain": float(surplus["teacher_gain"]),
        "probe_render_gain": float(surplus["render_gain"]),
        "has_surplus": bool(surplus["has_surplus"]),
    }


def is_structural_event(step: int, cfg) -> bool:
    """Mirror the ADC event gate without consulting any dataset metadata."""
    return bool(
        step < cfg.refine_stop_iter
        and step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    )


def save_visualization(
    path: Path, raw: Tensor, target: Tensor, prediction: Tensor, names: list[str]
) -> None:
    rows = min(4, raw.shape[0])
    h, w = raw.shape[-2:]
    canvas = Image.new("RGB", (3 * w, rows * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(rows):
        for column, image in enumerate((raw[row], target[row], prediction[row])):
            array = (
                image.permute(1, 2, 0).clamp(0, 1).numpy() * 255
            ).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (column * w, row * (h + 24)))
        draw.text((4, row * (h + 24) + h + 4), f"{names[row]} RAW", fill="black")
        draw.text((w + 4, row * (h + 24) + h + 4), "EVSSM/target", fill="black")
        draw.text(
            (2 * w + 4, row * (h + 24) + h + 4),
            "Learn2Splat sharp render",
            fill="black",
        )
    canvas.save(path)


def blur_kernel_statistics(
    objective: BlurAwareObjective,
    names: list[str],
    known_sharp: Tensor,
) -> tuple[Tensor, list[dict[str, object]]]:
    """Return interpretable per-view BPN kernel diagnostics."""
    with torch.no_grad():
        indices = torch.arange(
            objective.bpn.camera_embedding.num_embeddings,
            device=objective.bpn.camera_embedding.weight.device,
        )
        family = objective.bpn.kernel_family(indices)
        kernels = family["raw_kernels"].view(
            -1, objective.cfg.kernel_size, objective.cfg.kernel_size
        ).cpu()
        teacher_strength = family["teacher_strength"].cpu()
        raw_strength = family["raw_strength"].cpu()
    sharp = known_sharp.detach().bool().cpu().flatten()
    if kernels.shape[0] != len(names) or kernels.shape[0] != sharp.numel():
        raise ValueError(
            "BPN kernel/name/sharp lengths differ: "
            f"{kernels.shape[0]}, {len(names)}, {sharp.numel()}"
        )

    axis = torch.arange(objective.cfg.kernel_size, dtype=kernels.dtype)
    axis -= objective.cfg.kernel_size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dilation = float(objective.cfg.kernel_dilation)
    entropy_denominator = math.log(objective.cfg.kernel_size**2)
    last_mask_mean = objective.last_diagnostics.get("mask_mean")
    rows: list[dict[str, object]] = []
    for index, kernel in enumerate(kernels):
        center_x = float((kernel * xx).sum()) * dilation
        center_y = float((kernel * yy).sum()) * dilation
        radius = float(
            torch.sqrt((kernel * (xx.square() + yy.square())).sum())
        ) * dilation
        entropy = float(
            -(kernel * kernel.clamp_min(1e-12).log()).sum() / entropy_denominator
        )
        rows.append(
            {
                "index": index,
                "name": names[index],
                "known_sharp": bool(sharp[index]),
                "teacher_strength": float(teacher_strength[index]),
                "raw_strength": float(raw_strength[index]),
                "rms_radius_px": radius,
                "center_shift_px": math.hypot(center_x, center_y),
                "normalized_entropy": entropy,
                "center_mass": float(
                    kernel[
                        objective.cfg.kernel_size // 2,
                        objective.cfg.kernel_size // 2,
                    ]
                ),
                "peak_mass": float(kernel.max()),
                "last_batch_mask_mean": (
                    "" if last_mask_mean is None else float(last_mask_mean)
                ),
            }
        )
    return kernels, rows


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def stratified_kernel_indices(
    rows: list[dict[str, object]], limit: int = 16
) -> list[int]:
    """Select both sharp and non-sharp views instead of the first N cameras."""
    if limit <= 0:
        return []
    sharp = [int(row["index"]) for row in rows if bool(row["known_sharp"])]
    non_sharp = [int(row["index"]) for row in rows if not bool(row["known_sharp"])]
    sharp_count = min(len(sharp), limit // 2)
    non_sharp_count = min(len(non_sharp), limit - sharp_count)
    remaining = limit - sharp_count - non_sharp_count
    if remaining:
        sharp_count += min(remaining, len(sharp) - sharp_count)
        remaining = limit - sharp_count - non_sharp_count
        non_sharp_count += min(remaining, len(non_sharp) - non_sharp_count)
    return _evenly_spaced(sharp, sharp_count) + _evenly_spaced(
        non_sharp, non_sharp_count
    )


def save_kernel_visualization(
    path: Path,
    objective: BlurAwareObjective,
    names: list[str],
    known_sharp: Tensor,
    stats_path: Path | None = None,
) -> None:
    kernels, rows = blur_kernel_statistics(objective, names, known_sharp)
    if stats_path is not None:
        with stats_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    selected = stratified_kernel_indices(rows)
    columns = 4
    tile_side, label_height, header_height = 180, 34, 30
    row_count = max(1, math.ceil(len(selected) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_side, header_height + row_count * (tile_side + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    sharp_count = sum(bool(row["known_sharp"]) for row in rows)
    draw.text(
        (6, 7),
        f"BPN kernels: sharp={sharp_count}, non-sharp={len(rows) - sharp_count}",
        fill="black",
    )
    for slot, index in enumerate(selected):
        row = rows[index]
        normalized = kernels[index] / kernels[index].max().clamp_min(1e-8)
        heat = np.sqrt(normalized.numpy())
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        tile = Image.fromarray(heat).resize(
            (tile_side, tile_side), Image.Resampling.NEAREST
        )
        x = (slot % columns) * tile_side
        y = header_height + (slot // columns) * (tile_side + label_height)
        canvas.paste(tile, (x, y))
        kind = "S" if bool(row["known_sharp"]) else "B"
        draw.text(
            (x + 3, y + tile_side + 2),
            f"{kind} {row['name']} r={float(row['rms_radius_px']):.2f}",
            fill="black",
        )
        draw.text(
            (x + 3, y + tile_side + 16),
            f"H={float(row['normalized_entropy']):.3f} shift={float(row['center_shift_px']):.2f}",
            fill="black",
        )
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.adc in {"legs", "legs_blur"} and args.decoder_backend != "fastgs":
        raise ValueError(
            "LeGS-based ADC requires --decoder-backend fastgs; the official "
            "per-Gaussian leave-one-out sensitivity is a FastGS CUDA kernel"
        )
    if args.adc == "legs" and args.densification_reward != "off":
        raise ValueError(
            "exact LeGS uses its own delayed per-Gaussian sensitivity reward; "
            "do not combine it with the adapted global probe reward"
        )
    if args.adc == "legs_blur" and (
        args.objective != "blur-aware"
        or args.laplacian_loss_mode != "surplus"
        or args.densification_reward != "off"
    ):
        raise ValueError(
            "legs_blur requires --objective blur-aware, "
            "--laplacian-loss-mode surplus, and --densification-reward off; "
            "its delayed blur reward is internal to the LeGS policy"
        )
    reward_enabled = args.densification_reward == "surplus_probe"
    probe_enabled = args.densification_reward != "off"
    if probe_enabled and (
        args.objective != "blur-aware"
        or args.laplacian_loss_mode != "surplus"
        or args.adc != "adaptive"
        or args.optimizer not in {"learned", "learned_projected"}
    ):
        raise ValueError(
            "surplus_probe/probe_control requires --objective blur-aware, "
            "--laplacian-loss-mode surplus, --adc adaptive, and "
            "--optimizer learned or learned_projected"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")

    scene_configs = json.loads(Path(args.scene_config).read_text())
    if args.scene not in scene_configs:
        raise KeyError(f"unknown scene {args.scene}; choose from {sorted(scene_configs)}")
    cfg = scene_configs[args.scene]
    output_dir = Path(args.output_root) / args.scene / args.objective
    output_dir.mkdir(parents=True, exist_ok=False)

    parser = Parser(
        cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False
    )
    hold = read_hold(Path(cfg["data_dir"]))
    (
        optimization_indices,
        evaluation_indices,
        evaluation_source,
        frame_protocol,
    ) = resolve_scene_indices(
        parser, cfg, Path(cfg["data_dir"]), hold
    )
    scene = collect_scene(parser, cfg, evaluation_indices)
    scene_scale = float(parser.scene_scale * 1.1)
    evaluation_set = set(evaluation_indices)
    non_evaluation_indices = [
        index for index in optimization_indices if index not in evaluation_set
    ]
    if not non_evaluation_indices:
        non_evaluation_indices = optimization_indices
    probe_local = farthest_probe_indices(
        scene["c2w"][torch.tensor(non_evaluation_indices)], args.probe_views
    )
    probe_global = [non_evaluation_indices[index] for index in probe_local]
    train_views = build_views(
        scene,
        optimization_indices,
        scene_scale,
        device,
        policy_probe_indices=set(probe_global),
    )
    test_views = build_views(scene, evaluation_indices, scene_scale, device)

    requested_batch_strategy = args.opt_batch_strategy
    optgs = OptGS(
        checkpoint=args.checkpoint,
        device=device,
        decoder_backend=(
            None if args.decoder_backend == "checkpoint" else args.decoder_backend
        ),
        num_refine=(None if args.steps == 0 else args.steps),
        opt_batch_size=args.opt_batch_size,
        opt_batch_strategy=(
            None if requested_batch_strategy == "checkpoint" else requested_batch_strategy
        ),
        rollout_horizon=(None if args.rollout_horizon < 0 else args.rollout_horizon),
    )
    if optgs.checkpoint_num_refine is None:
        raise RuntimeError("checkpoint does not declare scene_trainer.num_update_steps")
    optimization_selection = torch.tensor(optimization_indices, dtype=torch.long)
    optimization_mass = scene["sampling_mass"][optimization_selection]
    sampler_realizes_risk = optgs.opt_batch_strategy == "supervision_fps"
    num_steps, update_budget_source = resolve_update_budget(
        args.steps,
        optgs.checkpoint_num_refine,
        len(optimization_indices),
        optgs.reference_context_views,
        optimization_mass if sampler_realizes_risk else None,
    )
    optgs.num_refine = num_steps
    optgs.configure_adc(args.adc, reward_conditioned=reward_enabled)
    configure_legs_blur_ablation(optgs.optimizer.cfg.refiner, args)
    init_budget, init_budget_source = resolve_initialization_budget(
        args.num_init_points, optgs
    )
    if args.initial_ply is None:
        gaussians, init_stats = depth_fused_initialization(
            parser,
            scene,
            cfg,
            optimization_indices,
            target_count=init_budget,
            max_sfm_points=args.max_sfm_points,
            sh_degree=optgs.sh_degree,
            device=device,
            seed=args.seed,
        )
    else:
        from optgs.experimental.api.integration.inria_bridge import (
            optgs_gaussians_from_ply,
        )

        initial_ply = Path(args.initial_ply).resolve()
        if not initial_ply.is_file():
            raise FileNotFoundError(f"continuation PLY does not exist: {initial_ply}")
        gaussians = optgs_gaussians_from_ply(
            initial_ply,
            sh_degree=optgs.sh_degree,
            device=device,
            dtype=optgs.dtype,
        )
        init_stats = {
            "source": "existing_ply_continuation",
            "path": str(initial_ply),
            "total": int(gaussians.means.shape[1]),
        }
    objective = None
    initial_objective_stats = None
    if args.objective == "blur-aware":
        objective = BlurAwareObjective(
            len(optimization_indices),
            BlurAwareObjectiveConfig(
                sharp_weight_in_sampler=(
                    requested_batch_strategy == "supervision_fps"
                ),
                laplacian_loss_weight=args.laplacian_loss_weight,
                laplacian_loss_mode=args.laplacian_loss_mode,
                coupled_dual_bpn=args.coupled_dual_bpn,
            ),
            known_sharp_mask=scene["known_sharp"][optimization_selection],
        )
        optgs.configure_input_objective(objective)
        if args.initial_objective_state is not None:
            if args.initial_ply is None:
                raise ValueError("--initial-objective-state requires --initial-ply")
            objective_path = Path(args.initial_objective_state).resolve()
            if not objective_path.is_file():
                raise FileNotFoundError(
                    f"objective continuation state does not exist: {objective_path}"
                )
            objective_payload = torch.load(
                objective_path, map_location=device, weights_only=True
            )
            saved_config = objective_payload.get("config")
            if saved_config != objective.export_config():
                raise RuntimeError(
                    "objective continuation config differs from the requested run"
                )
            objective.load_state_dict(objective_payload["model"], strict=True)
            initial_objective_stats = {
                "source": "existing_blur_aware_objective",
                "path": str(objective_path),
                "optimizer_state_restored": False,
            }
    optgs.initialize_from_tensors(gaussians, train_views)

    lpips_model = None
    if not args.skip_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()

    optimizer_override = None
    fallback_optimizer = None
    optimizer_switch_step = None
    if args.optimizer == "adam":
        from optgs.experimental.api.integration.config_bridge import build_adam_baseline

        optimizer_override = build_adam_baseline(num_steps, adc=args.adc).to(device)
        configure_legs_blur_ablation(optimizer_override.cfg.refiner, args)
    elif args.optimizer == "learned_projected" and num_steps > optgs.checkpoint_num_refine:
        from optgs.experimental.api.integration.config_bridge import build_adam_baseline

        optimizer_switch_step = optgs.checkpoint_num_refine
        fallback_optimizer = build_adam_baseline(
            num_steps,
            adc=args.adc,
            reward_conditioned=reward_enabled,
        ).to(device)
        configure_legs_blur_ablation(fallback_optimizer.cfg.refiner, args)

    probe_views = build_views(scene, probe_global, scene_scale, device)
    optimization_names = [
        Path(parser.image_names[index]).stem for index in optimization_indices
    ]
    optimization_sharp = scene["known_sharp"][
        torch.tensor(optimization_indices, dtype=torch.long)
    ]
    eval_steps = resolve_eval_steps(args.eval_steps, num_steps)
    metrics, diagnostics = [], []
    densification_probe_history: list[dict[str, float | int | bool]] = []
    refiner_cfg = optgs.optimizer.cfg.refiner
    final_gaussians = None
    for step, refined in optgs.optimize_iter(
        optimizer=optimizer_override,
        fallback_optimizer=fallback_optimizer,
        switch_step=optimizer_switch_step,
    ):
        iteration = step + 1
        final_gaussians = refined
        if objective is not None:
            diagnostics.append(dict(objective.last_diagnostics))
        next_step = step + 1
        if (
            probe_enabled
            and objective is not None
            and is_structural_event(next_step, refiner_cfg)
        ):
            feedback = render_densification_probe(
                optgs, refined, probe_views, objective
            )
            if reward_enabled:
                objective.set_densification_feedback(
                    probe_psnr=float(feedback["probe_psnr"]),
                    probe_surplus=float(feedback["probe_surplus"]),
                    has_surplus=bool(feedback["has_surplus"]),
                )
            densification_probe_history.append(
                {
                    "measured_after_updates": iteration,
                    "consumed_at_adc_step": next_step,
                    **feedback,
                }
            )
        if iteration in eval_steps:
            hold_metrics = render_metrics(
                optgs,
                refined,
                test_views,
                lpips_model=lpips_model,
                metric_batch_size=args.metric_batch_size,
            )
            metrics.append(
                {
                    "step": iteration,
                    "hold_psnr": hold_metrics["psnr"],
                    "hold_ssim": hold_metrics["ssim"],
                    "hold_lpips": hold_metrics["lpips"],
                    "hold_per_view_psnr": hold_metrics["per_view_psnr"],
                    "hold_per_view_lpips": hold_metrics["per_view_lpips"],
                    "valid_fraction": hold_metrics["valid_fraction"],
                    "num_gaussians": int(refined.means.shape[1]),
                }
            )
            save_visualization(
                output_dir / f"hold_step_{iteration:04d}.png",
                scene["raw_images"][torch.tensor(evaluation_indices)],
                scene["target_images"][torch.tensor(evaluation_indices)],
                hold_metrics["prediction"],
                [Path(parser.image_names[index]).stem for index in evaluation_indices],
            )
            if objective is not None:
                save_kernel_visualization(
                    output_dir / f"bpn_kernels_step_{iteration:04d}.png",
                    objective,
                    optimization_names,
                    optimization_sharp,
                    output_dir / f"bpn_kernel_stats_step_{iteration:04d}.csv",
                )

    if final_gaussians is None:
        raise RuntimeError("optimizer produced no steps")
    optgs.export_ply(str(output_dir / "point_cloud.ply"))
    if objective is not None:
        torch.save(
            {
                "model": objective.state_dict(),
                "optimizer": objective.optimizer_state_dict(),
                "config": objective.export_config(),
            },
            output_dir / "blur_aware_objective.pt",
        )

    reliability_rows = []
    for index, image_name in enumerate(parser.image_names):
        row = {
            "index": index,
            "name": Path(image_name).stem,
            "split": "eval+train" if index in evaluation_indices else "train",
            "known_sharp": bool(scene["known_sharp"][index]),
            "direct_supervision": bool(scene["direct_supervision"][index]),
            "auxiliary_supervision": bool(scene["auxiliary_mask"][index]),
            "supervision_confidence": float(
                scene["supervision_confidence"][index]
            ),
            "raw_path": scene["raw_paths"][index],
            "target_path": scene["target_paths"][index],
        }
        for key, values in scene["reliability"].items():
            row[key] = float(values[index])
        reliability_rows.append(row)
    with (output_dir / "reliability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reliability_rows[0].keys())
        writer.writeheader()
        writer.writerows(reliability_rows)
    if diagnostics:
        with (output_dir / "training_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=diagnostics[0].keys())
            writer.writeheader()
            writer.writerows(diagnostics)

    train_selection = torch.tensor(optimization_indices, dtype=torch.long)
    train_mass = scene["sampling_mass"][train_selection]
    train_sharp = scene["known_sharp"][train_selection]
    target_sharp_fraction = float(train_mass[train_sharp].sum() / train_mass.sum())
    sampler_draws = int(getattr(optgs, "_supervision_draws", 0))
    sampler_sharp_draws = int(getattr(optgs, "_supervision_sharp_draws", 0))

    receipt = {
        "scene": args.scene,
        "dataset": cfg["dataset"],
        "method": (
            (
                "official Learn2Splat Dense + exact LeGS per-Gaussian PPO "
                "+ factorized BPN + calibrated EVSSM + normalized NIMA-sharp w10"
                if args.adc == "legs"
                else (
                    "official Learn2Splat Dense + blur-conditioned LeGS PPO "
                    "+ factorized BPN + calibrated EVSSM + normalized NIMA-sharp w10"
                    if args.adc == "legs_blur"
                    else "official Learn2Splat Dense + factorized BPN + calibrated EVSSM "
                    "+ normalized NIMA-sharp w10"
                )
            )
            if args.optimizer == "learned"
            else (
                (
                    "official Learn2Splat Dense proposal + objective-consistent "
                    "Adam residual projection + exact LeGS per-Gaussian PPO"
                    if args.adc == "legs"
                    else (
                        "official Learn2Splat Dense proposal + objective-consistent "
                        "Adam residual projection + blur-conditioned LeGS PPO"
                        if args.adc == "legs_blur"
                        else "official Learn2Splat Dense proposal + objective-consistent "
                        "Adam residual projection + blur-aware capacity controller"
                    )
                )
                if args.optimizer == "learned_projected"
                else "official OptGS Adam diagnostic"
            )
        ),
        "same_config_contract": {
            "steps_requested": args.steps,
            "steps_effective": num_steps,
            "steps_source": update_budget_source,
            "checkpoint_steps": optgs.checkpoint_num_refine,
            "checkpoint_reference_context_views": optgs.reference_context_views,
            "checkpoint_reference_initial_gaussians": (
                optgs.reference_initial_gaussians
            ),
            "learned_rollout_horizon": optgs.rollout_horizon,
            "supervision_equivalent_views": (
                float(optimization_mass.sum() / optimization_mass.max())
                if sampler_realizes_risk
                else float(len(optimization_indices))
            ),
            "num_init_points_requested": args.num_init_points,
            "num_init_points_effective": init_budget,
            "num_init_points_source": init_budget_source,
            "initial_ply": args.initial_ply,
            "initial_objective_state": args.initial_objective_state,
            "opt_batch_size": args.opt_batch_size,
            "opt_batch_strategy_requested": requested_batch_strategy,
            "opt_batch_strategy_effective": optgs.opt_batch_strategy,
            "adc": args.adc,
            "decoder_backend_requested": args.decoder_backend,
            "decoder_backend_effective": optgs.decoder.cfg.name,
            "densification_reward": args.densification_reward,
            "capacity_controller": (
                {
                    "version": (
                        "blur_conditioned_legs_v6_unbiased_warmup"
                        if args.adc == "legs_blur"
                        else "official_legs_8eb120b_exact_mechanism"
                    ),
                    "host_representation": "learn2splat_gaussians",
                    "sensitivity": "official_fastgs_leave_one_out_l1",
                    "state_dim": 18 if args.adc == "legs_blur" else 11,
                    "local_state_objective": (
                        "evssm_bpn+raw_bpn+laplacian_surplus"
                        if args.adc == "legs_blur"
                        and refiner_cfg.local_objective_conditioned
                        else "target_only"
                    ),
                    "state_views": 10,
                    "quality_probe_views": (
                        len(probe_global) if args.adc == "legs_blur" else None
                    ),
                    "quality_probe_policy": (
                        "fixed_farthest_training_views"
                        if args.adc == "legs_blur"
                        else None
                    ),
                    "blur_state": (
                        [
                            "evssm_reliability_mean",
                            "evssm_reliability_dispersion",
                            "laplacian_surplus",
                            "bpn_kernel_entropy",
                            "bpn_kernel_radius",
                            "bpn_mask_strength",
                            "primitive_pressure",
                        ]
                        if args.adc == "legs_blur"
                        else None
                    ),
                    "actions": ["keep", "clone", "split"],
                    "pruning": "separate_low_opacity_estimator",
                    "reward_delay": 50,
                    "reward": (
                        (
                            "normalized sensitivity + confidence-weighted teacher "
                            "PSNR delta + reliability-complement RAW-BPN PSNR delta "
                            "+ Laplacian-surplus delta - relative net capacity growth, "
                            "directionally assigned to birth/prune and soft-assigned "
                            "by local sensitivity support"
                            if args.coupled_dual_bpn
                            else
                            "normalized sensitivity + confidence-weighted PSNR delta "
                            "+ Laplacian-surplus delta - relative net capacity growth, "
                            "directionally assigned to birth/prune and soft-assigned "
                            "by local sensitivity support"
                        )
                        if args.adc == "legs_blur"
                        else "normalized delayed per-Gaussian sensitivity"
                    ),
                    "cross_scene_normalization": (
                        "dimensionless bounded state features and causal reward RMS"
                        if args.adc == "legs_blur"
                        else None
                    ),
                    "blur_conditioning": (
                        {
                            "adapter": "zero_initialized_residual",
                            "start_iter": int(
                                refiner_cfg.blur_condition_start_iter
                            ),
                            "ramp_iters": int(
                                refiner_cfg.blur_condition_ramp_iters
                            ),
                            "quality_weight": float(
                                refiner_cfg.blur_quality_weight
                            ),
                            "capacity_weight": float(
                                refiner_cfg.blur_capacity_weight
                            ),
                            "quality_gates_capacity_cost": True,
                            "credit_assignment": (
                                "net_action_direction_times_threshold_free_"
                                "sigmoid_of_standardized_local_delayed_sensitivity"
                            ),
                        }
                        if args.adc == "legs_blur"
                        else None
                    ),
                    "parent_child_credit": True,
                    "schedule": {
                        "start": 500,
                        "interval": 100,
                        "stop": 15000,
                        "opacity_reset": 3000,
                    },
                    "global_primitive_cap": None,
                }
                if args.adc in {"legs", "legs_blur"}
                else
                {
                    "version": (
                        "residual_headroom_v2"
                        if args.adc == "adaptive"
                        else "visibility_cap_v1"
                    ),
                    "reference_budget_source": "learn2splat_checkpoint",
                    "base_cap": "reference_active_budget / visible_fraction",
                    "demand_cap": (
                        "base_cap / (1 - residual_support_fraction)"
                        if args.adc == "adaptive"
                        else "base_cap"
                    ),
                    "safety_cap_role": "hardware_only",
                }
                if args.adc in {"adaptive", "adaptive_legacy"}
                else None
            ),
            "objective": args.objective,
            "optimizer": args.optimizer,
            "optimizer_switch_step": optimizer_switch_step,
            "optimizer_transitions": optgs.optimizer_transitions,
            "seed": args.seed,
        },
        "hold": hold,
        "protocol": cfg.get(
            "protocol_description",
            (
                "resolved input stream optimized; resolved sharp subset evaluated"
                if frame_protocol is not None
                else (
                    "all input frames optimized; explicit original-frame subset evaluated"
                    if evaluation_source.startswith("manifest:")
                    else (
                        "all input frames optimized; authoritative sharp subset evaluated"
                        if evaluation_source == "authoritative_scene_manifest"
                        else "all input frames optimized; hold subset evaluated"
                    )
                )
            ),
        ),
        "benchmark_role": cfg.get("benchmark_role", "benchmark"),
        "optimization_indices": optimization_indices,
        "evaluation_indices": evaluation_indices,
        "evaluation_source": evaluation_source,
        "evaluation_reference": "authoritative_raw_image",
        "frame_protocol": frame_protocol,
        "probe_train_indices": probe_global,
        "densification_probe_history": densification_probe_history,
        "initialization": init_stats,
        "objective_initialization": initial_objective_stats,
        "camera_preprocess": [
            {
                "camera_id": int(camera_id),
                "source_size": preprocessor.source_size,
                "pre_crop_left_top_right_bottom": preprocessor.pre_crop,
                "resize_size": preprocessor.resize_size,
                "crop_left_top_right_bottom": preprocessor.crop,
                "output_size": preprocessor.output_size,
                "distortion": preprocessor.distortion,
                "undistort_depth": preprocessor.undistort_depth,
            }
            for camera_id, preprocessor in {
                int(camera_id): preprocessor
                for camera_id, preprocessor in zip(
                    parser.camera_ids, scene["camera_preprocessors"]
                )
            }.items()
        ],
        "sharp_supervision": {
            "count": int(scene["known_sharp"].sum()),
            "sources": scene["sharp_sources"],
            "policy": scene["sharp_policy"],
            "relative_weight": 10.0,
            "risk_implementation": (
                "scene_exposure_distribution"
                if requested_batch_strategy == "supervision_fps"
                else "per_batch_loss_weight"
            ),
            "target_exposure_fraction": target_sharp_fraction,
            "actual_exposure_fraction": (
                None
                if sampler_draws == 0
                else sampler_sharp_draws / sampler_draws
            ),
            "sampling_draws": sampler_draws,
        },
        "auxiliary_supervision": {
            "count": int(scene["auxiliary_mask"].sum()),
            "sources": scene["auxiliary_sources"],
            "direct_target": True,
            "receives_sharp_w10": False,
            "confidence_by_name": {
                Path(parser.image_names[index]).stem: float(
                    scene["supervision_confidence"][index]
                )
                for index in range(len(parser.image_names))
                if bool(scene["auxiliary_mask"][index])
            },
        },
        "objective_config": None if objective is None else objective.export_config(),
        "metrics_config": {
            "psnr": True,
            "ssim": True,
            "lpips": not args.skip_lpips,
            "lpips_network": None if args.skip_lpips else "alexnet-v0.1",
            "metric_batch_size": args.metric_batch_size,
        },
        "capacity_events": optgs.capacity_events,
        "metrics": metrics,
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
