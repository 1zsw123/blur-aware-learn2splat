#!/usr/bin/env python3
"""Read-only cross-scene diagnostics for the PRISM3D ExBlur experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from optgs.dataset.colmap.utils import Parser


SCENES = (
    "bench",
    "camellia",
    "dragon",
    "jars",
    "jars2",
    "postbox",
    "stone_lantern",
    "sunflowers",
)
PROBLEM_SCENES = {"camellia", "jars", "jars2", "stone_lantern"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--run-10k-root", required=True)
    parser.add_argument("--run-50k-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scenes",
        default=",".join(SCENES),
        help="comma-separated scene subset, used to bound diagnostic memory",
    )
    parser.add_argument("--skip-longitudinal", action="store_true")
    return parser.parse_args()


def image_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def image_path(directory: Path, stem: str) -> Path:
    matches = sorted(directory.glob(f"{stem}.*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one image for {stem} under {directory}: {matches}")
    return matches[0]


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean(np.square(left - right)))
    return -10.0 * math.log10(max(mse, 1e-10))


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def orb_ready(image: np.ndarray) -> np.ndarray:
    value = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 640.0 / value.shape[1])
    if scale < 1.0:
        value = cv2.resize(value, None, fx=scale, fy=scale)
    return value


def orb_pair(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    detector = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
    key_left, desc_left = detector.detectAndCompute(left, None)
    key_right, desc_right = detector.detectAndCompute(right, None)
    if desc_left is None or desc_right is None:
        return len(key_left), 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 8:
        return len(key_left), len(good), 0.0
    source = np.float32([key_left[item.queryIdx].pt for item in good])
    target = np.float32([key_right[item.trainIdx].pt for item in good])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inlier_fraction = float(mask.mean()) if mask is not None else 0.0
    return len(key_left), len(good), inlier_fraction


def quantiles(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p10": float(np.quantile(array, 0.10)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
    }


def scene_data_metrics(cfg: dict, parser: Parser) -> dict[str, float]:
    raw_dir = Path(cfg["raw_dir"])
    turtle_dir = Path(cfg["evssm_dir"])
    sharp_dir = raw_dir.parent / "images_sharp"
    raw_psnr_values = []
    turtle_psnr_values = []
    raw_lap_ratio = []
    turtle_lap_ratio = []
    raw_images = []
    sharp_images = []
    for name in parser.image_names:
        stem = Path(name).stem
        sharp = image_array(image_path(sharp_dir, stem))
        size = (sharp.shape[1], sharp.shape[0])
        raw = image_array(image_path(raw_dir, stem), size)
        turtle = image_array(image_path(turtle_dir, stem), size)
        sharp_lap = max(laplacian_variance(sharp), 1e-8)
        raw_psnr_values.append(psnr(raw, sharp))
        turtle_psnr_values.append(psnr(turtle, sharp))
        raw_lap_ratio.append(laplacian_variance(raw) / sharp_lap)
        turtle_lap_ratio.append(laplacian_variance(turtle) / sharp_lap)
        raw_images.append(orb_ready(raw))
        sharp_images.append(orb_ready(sharp))

    translation = []
    rotation = []
    raw_keypoints = []
    raw_matches = []
    raw_inliers = []
    sharp_keypoints = []
    sharp_matches = []
    sharp_inliers = []
    for index in range(len(parser.image_names) - 1):
        left_pose = parser.camtoworlds[index]
        right_pose = parser.camtoworlds[index + 1]
        translation.append(
            float(
                np.linalg.norm(left_pose[:3, 3] - right_pose[:3, 3])
                / max(float(parser.scene_scale), 1e-8)
            )
        )
        rotation.append(rotation_angle_degrees(left_pose, right_pose))
        keypoints, matches, inliers = orb_pair(raw_images[index], raw_images[index + 1])
        raw_keypoints.append(float(keypoints))
        raw_matches.append(float(matches))
        raw_inliers.append(inliers)
        keypoints, matches, inliers = orb_pair(
            sharp_images[index], sharp_images[index + 1]
        )
        sharp_keypoints.append(float(keypoints))
        sharp_matches.append(float(matches))
        sharp_inliers.append(inliers)

    result = {
        "frames": len(parser.image_names),
        "sfm_points": int(len(parser.points)),
        "sfm_error_median": float(np.median(parser.points_err)),
        "sfm_error_p90": float(np.quantile(parser.points_err, 0.90)),
        "turtle_psnr_gain_mean": float(
            np.mean(np.asarray(turtle_psnr_values) - np.asarray(raw_psnr_values))
        ),
    }
    for values, prefix in (
        (raw_psnr_values, "raw_sharp_psnr"),
        (turtle_psnr_values, "turtle_sharp_psnr"),
        (raw_lap_ratio, "raw_sharp_lap_ratio"),
        (turtle_lap_ratio, "turtle_sharp_lap_ratio"),
        (translation, "pose_translation"),
        (rotation, "pose_rotation_deg"),
        (raw_keypoints, "raw_orb_keypoints"),
        (raw_matches, "raw_orb_matches"),
        (raw_inliers, "raw_orb_inlier_fraction"),
        (sharp_keypoints, "sharp_orb_keypoints"),
        (sharp_matches, "sharp_orb_matches"),
        (sharp_inliers, "sharp_orb_inlier_fraction"),
    ):
        result.update(quantiles(values, prefix))
    return result


def bpn_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    result = {"bpn_views": len(rows)}
    for column in (
        "teacher_strength",
        "raw_strength",
        "rms_radius_px",
        "center_shift_px",
        "normalized_entropy",
        "center_mass",
        "peak_mass",
        "last_batch_mask_mean",
    ):
        result.update(quantiles([float(row[column]) for row in rows], f"bpn_{column}"))
    result["bpn_radius_gt8_fraction"] = float(
        np.mean([float(row["rms_radius_px"]) > 8.0 for row in rows])
    )
    result["bpn_radius_lt1_fraction"] = float(
        np.mean([float(row["rms_radius_px"]) < 1.0 for row in rows])
    )
    return result


def diagnostics_windows(path: Path) -> list[dict[str, float]]:
    frame = pd.read_csv(path)
    columns = (
        "loss",
        "direct_loss",
        "raw_loss",
        "kernel_entropy",
        "mask_mean",
        "laplacian_loss",
        "laplacian_teacher_gain",
        "laplacian_render_gain",
        "laplacian_relative_gain",
        "laplacian_floor",
        "laplacian_overshoot",
        "laplacian_artifact",
        "static_confidence",
        "effective_confidence",
        "dynamic_uncertainty",
        "surplus_consensus",
    )
    output = []
    maximum = int(frame["step"].max())
    for start in range(0, maximum + 1, 10000):
        window = frame[(frame["step"] >= start) & (frame["step"] < start + 10000)]
        if window.empty:
            continue
        record = {"start": start, "end": start + 10000}
        for column in columns:
            record[f"{column}_mean"] = float(window[column].mean())
            record[f"{column}_last"] = float(window[column].iloc[-1])
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    configs = json.loads(Path(args.scene_config).read_text())
    root_10k = Path(args.run_10k_root)
    root_50k = Path(args.run_50k_root)
    report = {"scenes": {}, "longitudinal": {}}
    selected_scenes = tuple(value.strip() for value in args.scenes.split(","))
    unknown = set(selected_scenes) - set(SCENES)
    if unknown:
        raise ValueError(f"unknown scenes: {sorted(unknown)}")
    for scene in selected_scenes:
        key = f"prism3d_{scene}_turtle"
        cfg = configs[key]
        parser = Parser(
            cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False
        )
        run_dir = root_10k / key / "blur-aware"
        receipt = json.loads((run_dir / "receipt.json").read_text())
        record = {
            "group": "problem" if scene in PROBLEM_SCENES else "control",
            **scene_data_metrics(cfg, parser),
            **bpn_metrics(run_dir / "bpn_kernel_stats_step_10000.csv"),
            "hold_metrics_10k": receipt["metrics"][0],
            "initialization": receipt["initialization"],
            "sharp_supervision": receipt["sharp_supervision"],
            "capacity_last_10k": receipt["capacity_events"][-1],
        }
        report["scenes"][scene] = record

    for scene in (
        value
        for value in ("jars", "jars2")
        if value in selected_scenes and not args.skip_longitudinal
    ):
        key = f"prism3d_{scene}_turtle"
        run_dir = root_50k / key / "blur-aware"
        receipt = json.loads((run_dir / "receipt.json").read_text())
        report["longitudinal"][scene] = {
            "metrics": receipt["metrics"],
            "capacity_events": receipt["capacity_events"],
            "diagnostic_windows": diagnostics_windows(
                run_dir / "training_diagnostics.csv"
            ),
            "bpn": {
                str(step): bpn_metrics(
                    run_dir / f"bpn_kernel_stats_step_{step}.csv"
                )
                for step in (10000, 20000, 30000, 40000, 50000)
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
