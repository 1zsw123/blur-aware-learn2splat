#!/usr/bin/env python3
"""Summarize VGGSfM track support and adjacent-view overlap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_colmap_loader():
    path = Path(__file__).parents[2] / "third_party/LeGS/scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("legs_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def point_track_lengths(path: Path) -> np.ndarray:
    lengths = []
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            handle.read(43)
            length = struct.unpack("<Q", handle.read(8))[0]
            lengths.append(length)
            handle.seek(8 * length, 1)
    return np.asarray(lengths, dtype=np.float64)


def summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.1)),
        f"{prefix}_p90": float(np.quantile(values, 0.9)),
    }


def main() -> None:
    args = parse_args()
    configs = json.loads(Path(args.scene_config).read_text())
    loader = load_colmap_loader()
    report = {}
    for key, cfg in configs.items():
        scene = key.removeprefix("prism3d_").removesuffix("_turtle")
        sparse = Path(cfg["data_dir"]) / "sparse/0"
        images = loader.read_extrinsics_binary(str(sparse / "images.bin"))
        ordered = sorted(images.values(), key=lambda image: Path(image.name).stem)
        observations = []
        point_sets = []
        for image in ordered:
            points = {int(value) for value in image.point3D_ids if int(value) >= 0}
            point_sets.append(points)
            observations.append(float(len(points)))
        shared = []
        overlap_min = []
        jaccard = []
        for left, right in zip(point_sets, point_sets[1:]):
            intersection = len(left & right)
            shared.append(float(intersection))
            overlap_min.append(intersection / max(min(len(left), len(right)), 1))
            jaccard.append(intersection / max(len(left | right), 1))
        tracks = point_track_lengths(sparse / "points3D.bin")
        report[scene] = {
            "frames": len(ordered),
            "points": int(len(tracks)),
            **summary(np.asarray(observations), "observations_per_image"),
            **summary(np.asarray(shared), "adjacent_shared_points"),
            **summary(np.asarray(overlap_min), "adjacent_overlap_min"),
            **summary(np.asarray(jaccard), "adjacent_jaccard"),
            **summary(tracks, "track_length"),
            "track_ge3_fraction": float(np.mean(tracks >= 3)),
            "track_ge5_fraction": float(np.mean(tracks >= 5)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
