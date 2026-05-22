import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np
import os
import random

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .data_types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    # random crop training
    random_crop: bool
    max_size: list[int] | None
    min_size: list[int] | None

    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    no_mix_test_set: bool = True
    load_depth: bool = False
    min_views: int = 0
    max_views: int = 0
    highres: bool = False
    sort_target_index: Optional[bool] = False
    overfit_max_views: Optional[int] = None
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False
    pose_align_first_view: bool = False  # align the camera pose to the first view
    scale_extrinsics: float = 1.
    metric_scale_align_dl3dv: bool = False
    center_pose: bool = False  # center and normalize the pose by the distance to the center

    # mix re10k & dl3dv
    mix_re10k: bool = False
    re10k_min_view_dist: int = 40
    re10k_max_view_dist: int = 300

    # load remaining context views
    load_remain_context: bool = False
    num_remain_context: int = 8

    index_name: str = "index.json"


class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
            self,
            cfg: DatasetDL3DVCfg,
            stage: Stage,
            view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / self.cfg.index_name, "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            # mixed data training only evaluate on a single test set
            if cfg.no_mix_test_set and self.data_stage in ['val', 'test'] and i > 0:
                continue

            # balance the datasets for mixed dataset training
            # for gs: mix re10k, dl3dv
            if len(cfg.roots) > 1 and self.data_stage == 'train':
                if 'dl3dv' in str(root):
                    root_chunks = root_chunks * 8

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # fast testing
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))

        if self.cfg.metric_scale_align_dl3dv:
            # read invalid scales
            scale_dir = '/cluster/project/cvg/haofei/datasets/depthsplat/dl3dv_metric_scale_factor'
            filename = os.path.join(scale_dir, 'dl3dv_invalid.txt')
            with open(filename, "r") as f:
                self.invalid_scale_scenes = [line.strip() for line in f]

        # Calculate actual number of scenes (keys of index file that their chunks exist).
        num_available_scenes = 0
        for scene, chunk_path in self.index.items():
            if chunk_path in self.chunks:
                num_available_scenes += 1
        self.num_available_scenes = num_available_scenes

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def _get_scale_factor(self, scene: str) -> float:
        """Get the scale factor for a scene."""
        if self.cfg.metric_scale_align_dl3dv:
            scale_dir = '/cluster/project/cvg/haofei/datasets/depthsplat/dl3dv_metric_scale_factor'
            if self.stage == 'train':
                folder = scene.split('_')[1]
            else:
                folder = scene
            filename = os.path.join(scale_dir, folder, 'scale_factor.txt')

            if not os.path.exists(filename) or folder in self.invalid_scale_scenes:
                return self.cfg.scale_extrinsics
            else:
                with open(filename, "r") as f:
                    return float(f.read().strip())
        else:
            return self.cfg.scale_extrinsics

    def _process_example_to_batch(
            self,
            example: dict,
            extrinsics: Tensor,
            intrinsics: Tensor,
            context_indices: Tensor,
            target_indices: Tensor,
            scale_factor: float,
    ) -> Optional[dict]:
        """
        Process an example into a batch dict (original behavior).
        Returns None if the example should be skipped.
        """
        scene = example["key"]

        # Load remaining context views if configured
        if self.cfg.load_remain_context:
            remaining_indices = get_remaining_indices(
                context_indices, target_indices, self.cfg.num_remain_context
            )
            remain_context_images = [
                example["images"][index.item()] for index in remaining_indices
            ]
            try:
                remain_context_images = self.convert_images(remain_context_images)
            except OSError:
                return None

        # Load context images
        context_images = [
            example["images"][index.item()] for index in context_indices
        ]
        try:
            context_images = self.convert_images(context_images)
        except OSError:
            return None

        # Load target images
        target_images = [
            example["images"][index.item()] for index in target_indices
        ]
        try:
            target_images = self.convert_images(target_images)
        except OSError:
            return None

        # Validate image shapes
        if self.cfg.mix_re10k and 'dl3dv' not in scene:
            if self.cfg.highres:
                expected_shape = (3, 720, 1280)
            else:
                expected_shape = (3, 360, 640)
        else:
            expected_shape = tuple([3, *self.cfg.ori_image_shape])

        if self.stage in ['test', 'val'] or 'dl3dv' in scene:
            expected_shape = tuple([3, *self.cfg.ori_image_shape])

        if self.cfg.skip_bad_shape:
            if context_images.shape[1:] != expected_shape or target_images.shape[1:] != expected_shape:
                print(
                    f"Skipped bad example {scene}. Context shape was "
                    f"{context_images.shape}, target shape was "
                    f"{target_images.shape}, and expected shape was {expected_shape}"
                )
                return None

            if self.cfg.load_remain_context and remain_context_images.shape[1:] != expected_shape:
                return None

        # Apply pose transformations
        if self.cfg.pose_align_middle_view:
            mid_index = context_indices.shape[0] // 2
            extrinsics = camera_normalization(
                extrinsics[context_indices][mid_index:mid_index + 1], extrinsics
            )

        if self.cfg.pose_align_first_view:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if self.cfg.center_pose:
            extrinsics = center_norm_pose(extrinsics)

        # Validate extrinsics
        if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
            return None
        if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
            return None
        if (extrinsics[context_indices][:, :3, 3].abs() > 1e3).any():
            return None
        if (extrinsics[target_indices][:, :3, 3].abs() > 1e3).any():
            return None
        if not torch.allclose(
                torch.det(extrinsics[context_indices][:, :3, :3]),
                extrinsics[context_indices][:, :3, :3].new_tensor(1)
        ):
            return None
        if not torch.allclose(
                torch.det(extrinsics[target_indices][:, :3, :3]),
                extrinsics[target_indices][:, :3, :3].new_tensor(1)
        ):
            return None

        if self.cfg.load_remain_context:
            if any(torch.isnan(torch.det(extrinsics[remaining_indices][:, :3, :3]))):
                return None
            if (extrinsics[remaining_indices][:, :3, 3] > 1e3).any():
                return None
            if not torch.allclose(
                    torch.det(extrinsics[remaining_indices][:, :3, :3]),
                    extrinsics[remaining_indices][:, :3, :3].new_tensor(1)
            ):
                return None

        # Apply scale factor
        extrinsics[:, :3, 3] *= scale_factor

        # Build output
        example_out = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
            },
            "scene": scene,
        }

        if self.cfg.load_remain_context:
            example_out["context_remain"] = {
                "extrinsics": extrinsics[remaining_indices],
                "intrinsics": intrinsics[remaining_indices],
                "image": remain_context_images,
                "near": self.get_bound("near", len(remaining_indices)),
                "far": self.get_bound("far", len(remaining_indices)),
                "index": remaining_indices,
            }

        return example_out

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        # Counter for skipping the first test_start_idx scenes (used for scene-chunked SLURM jobs).
        test_scene_counter = 0

        # Iterate over chunks.
        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                if self.stage == "test":
                    chunk = item
                else:
                    chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            # Iterate over examples in the chunk.
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                if example["key"] not in self.index:
                    continue

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                # Skip if field of view is too wide
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                scale_factor = self._get_scale_factor(scene)

                try:
                    extra_kwargs = {}
                    if self.cfg.overfit_to_scene is not None and self.stage != "test":
                        extra_kwargs["max_num_views"] = (
                            148 if self.cfg.overfit_max_views is None
                            else self.cfg.overfit_max_views
                        )

                    is_re10k = self.cfg.mix_re10k and 'dl3dv' not in scene and self.stage == 'train'

                    out_data = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        min_context_views=self.cfg.min_views,
                        max_context_views=self.cfg.max_views,
                        min_view_dist=self.cfg.re10k_min_view_dist if is_re10k else None,
                        max_view_dist=self.cfg.re10k_max_view_dist if is_re10k else None,
                        **extra_kwargs,
                    )

                    if isinstance(out_data, tuple):
                        context_indices, target_indices = out_data[:2]
                        c_list = [
                            context_indices.sort()[0] if self.cfg.sort_context_index else context_indices
                        ]
                        t_list = [
                            target_indices.sort()[0] if self.cfg.sort_target_index else target_indices
                        ]
                    elif isinstance(out_data, list):
                        c_list = [
                            a.context.sort()[0] if self.cfg.sort_context_index else a.context
                            for a in out_data
                        ]
                        t_list = [
                            a.target.sort()[0] if self.cfg.sort_target_index else a.target
                            for a in out_data
                        ]

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    example_out = self._process_example_to_batch(
                        example, extrinsics.clone(), intrinsics,
                        context_indices, target_indices, scale_factor
                    )

                    if example_out is None:
                        continue

                    # Apply augmentation and cropping
                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)

                    # Skip scenes before test_start_idx (for scene-chunked SLURM jobs)
                    if self.stage == "test" and test_scene_counter < self.cfg.test_start_idx:
                        test_scene_counter += 1
                        continue

                    context_images = example_out["context"]["image"]
                    if self.cfg.image_shape == list(context_images.shape[2:]):
                        yield example_out
                    else:
                        if self.stage == "train" and self.cfg.random_crop:
                            crop_h = random.randint(self.cfg.min_size[0], self.cfg.max_size[0] + 1) // 64 * 64
                            crop_w = random.randint(self.cfg.min_size[1], self.cfg.max_size[1] + 1) // 64 * 64
                            crop_size = (crop_h, crop_w)
                            yield apply_crop_shim(example_out, crop_size)
                        else:
                            yield apply_crop_shim(example_out, tuple(self.cfg.image_shape))

    def convert_poses(
            self,
            poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

        if self.cfg.opencv_pose_format:
            return self.opengl_to_opencv(w2c.inverse()), intrinsics
        else:
            return w2c.inverse(), intrinsics

    def opengl_to_opencv(self, c2w):
        # https://github.com/DL3DV-10K/Dataset/issues/4#issuecomment-2019441741
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        blender2opencv = torch.tensor(blender2opencv, dtype=c2w.dtype, device=c2w.device).unsqueeze(0)
        c2w = torch.matmul(c2w, blender2opencv)
        c2w[:, 2, :] *= -1
        c2w = c2w[:, torch.tensor(np.array([1, 0, 2, 3])), :]
        c2w[:, 0:3, 1:3] *= -1

        return c2w

    def convert_images(
            self,
            images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
            self,
            bound: Literal["near", "far"],
            num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                with (root / data_stage / self.cfg.index_name).open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v)
                         for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # mixed data training only evaluate on a single test set
                if self.cfg.no_mix_test_set and data_stage == 'test' and i > 0:
                    continue

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        """Calculate dataset length based on stage and configuration."""
        if self.stage == "test":
            test_length = self.num_available_scenes * self.cfg.test_times_per_scene
            if self.cfg.test_len > 0:
                return min(test_length, self.cfg.test_len)
            else:
                return test_length
        elif self.stage == "train":
            return self.num_available_scenes * self.cfg.train_times_per_scene
        else:
            # Validation stage: return large value to prevent exhaustion
            # The actual length will be wrapped to 1 by the dataloader
            return int(1e10)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    # Manually calculate the inverse of SE(3) to avoid numerical issues
    R = pivotal_pose[:, :3, :3]  # [1, 3, 3]
    t = pivotal_pose[:, :3, 3:]  # [1, 3, 1]
    R_inv = R.transpose(-1, -2)  # [1, 3, 3]
    t_inv = -R_inv @ t  # [1, 3, 1]
    camera_norm_matrix_manuall = torch.eye(4, dtype=poses.dtype, device=poses.device).unsqueeze(0)  # [1, 4, 4]
    camera_norm_matrix_manuall[:, :3, :3] = R_inv
    camera_norm_matrix_manuall[:, :3, 3:] = t_inv

    # normalize all views
    normalized_poses = camera_norm_matrix_manuall @ poses  # [N, 4, 4]

    return normalized_poses


def center_norm_pose(extrinsics):
    # extrinsics: [V, 4, 4]
    cam_centers = extrinsics[:, :3, 3]  # [V, 3]
    avg_center = cam_centers.mean(dim=0, keepdim=True)  # [1, 3]
    dist = (cam_centers - avg_center).norm(dim=1, keepdim=True)  # [V, 1]
    scale = dist.max()

    # translate
    extrinsics = extrinsics.clone()
    extrinsics[:, :3, 3] -= avg_center
    extrinsics[:, :3, 3] /= scale

    return extrinsics


def get_remaining_indices(context_indices: torch.Tensor,
                          target_indices: torch.Tensor,
                          num_remain_context: int) -> torch.Tensor:
    """
    Randomly selects a fixed number of remaining indices in the range [min(context), max(context)],
    excluding those in context or target. Pads by repeating if not enough remain.

    Args:
        context_indices (torch.Tensor): 1D tensor of context indices.
        target_indices (torch.Tensor): 1D tensor of target indices.
        num_remain_context (int): Number of remaining indices to return.

    Returns:
        torch.Tensor: 1D tensor of length `num_remain_context`.
    """
    if context_indices.numel() == 0:
        raise ValueError("context_indices must not be empty.")

    min_idx = torch.min(context_indices).item()
    max_idx = torch.max(context_indices).item()

    full_range = torch.arange(min_idx, max_idx + 1, dtype=torch.long)
    exclude_indices = torch.cat([context_indices, target_indices])
    mask = ~torch.isin(full_range, exclude_indices)

    remaining = full_range[mask]

    if remaining.numel() == 0:
        # Nothing to sample from; repeat the first context index (or any fallback)
        return context_indices[0].repeat(num_remain_context)

    # return all
    selected = remaining

    # Randomly sample with or without replacement
    # if remaining.numel() >= num_remain_context:
    #     selected = remaining[torch.randperm(remaining.numel())[:num_remain_context]]
    # else:
    #     # return all
    #     selected = remaining
    #     # # Repeat with wrap-around to pad
    #     # num_repeat = (num_remain_context + remaining.numel() - 1) // remaining.numel()
    #     # padded = remaining.repeat(num_repeat)[:num_remain_context]
    #     # selected = padded[torch.randperm(num_remain_context)]  # Shuffle for randomness

    return selected.sort().values
