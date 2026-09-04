#!/usr/bin/env python3
"""Prepare immutable ExBlur split manifests and EVSSM input trees."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


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


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--evssm-root", type=Path, required=True)
    parser.add_argument("--nima-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    args = parser.parse_args()
    args.release_root = args.release_root.resolve()
    args.stage_root = args.stage_root.resolve()
    args.manifest_root = args.manifest_root.resolve()
    args.converted_root = args.converted_root.resolve()
    args.evssm_root = args.evssm_root.resolve()
    args.nima_root = args.nima_root.resolve()
    args.scene_config = args.scene_config.resolve()

    args.stage_root.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    scene_config = {}
    for scene in SCENES:
        source = args.release_root / scene
        image_files = sorted(
            path for path in (source / "images").iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        train_names = (source / "train.txt").read_text().split()
        test_indices = [int(value) for value in (source / "test.txt").read_text().split()]
        hold_files = list(source.glob("hold=*"))
        if len(hold_files) != 1:
            raise RuntimeError(f"{scene}: expected exactly one hold=N file")
        hold = int(hold_files[0].name.split("=", 1)[1])
        if train_names != [path.name for path in image_files]:
            raise RuntimeError(f"{scene}: train.txt does not bind the full ordered stream")
        expected = list(range(0, len(image_files), hold))
        if test_indices != expected:
            raise RuntimeError(f"{scene}: test indices {test_indices} != hold split {expected}")

        test_names = [image_files[index].name for index in test_indices]
        (args.manifest_root / f"{scene}_official_test.json").write_text(
            json.dumps(test_names, indent=2) + "\n"
        )
        input_dir = args.stage_root / scene / "test" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for image in image_files:
            safe_symlink(image.resolve(), input_dir / image.name)
        summary[scene] = {
            "images": len(image_files),
            "hold": hold,
            "test_indices": test_indices,
            "test_names": test_names,
        }
        sharp_json = args.nima_root / f"{scene}_exblur_nima06_sharp_frames.json"
        if not sharp_json.is_file():
            raise FileNotFoundError(sharp_json)
        scene_config[f"exblur_{scene}"] = {
            "dataset": "ExBlur-NeRF",
            "data_dir": str(args.converted_root / scene),
            "factor": 1,
            "raw_dir": str(source / "images"),
            "raw_mode": "stem",
            "evssm_dir": str(args.evssm_root / scene),
            "sharp_json": str(sharp_json),
            "sharp_supervision_policy": "sharp_json_only",
            "evaluation_direct_supervision": True,
            "exclude_evaluation_from_optimization": False,
            "require_sharp_evaluation_targets": True,
            "protocol_description": (
                "all ExBlur input views optimized; official hold views use "
                "authoritative direct sharp supervision and evaluation; w10 "
                "remains restricted to NIMA>0.6"
            ),
            "evaluation_manifest": {
                "path": str(args.manifest_root / f"{scene}_official_test.json"),
                "label": "unblurslam_exblur_official_test",
            },
        }
    (args.manifest_root / "exblur8_official_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    args.scene_config.write_text(json.dumps(scene_config, indent=2) + "\n")


if __name__ == "__main__":
    main()
