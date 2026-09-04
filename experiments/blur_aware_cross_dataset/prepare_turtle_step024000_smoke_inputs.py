#!/usr/bin/env python3
"""Build three immutable five-frame windows for Turtle checkpoint smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path


BASE = Path("/srv2/szha0669/blur_slam_exp")
HERE = BASE / "repos/learn2splat-official-space/experiments/blur_aware_cross_dataset"
STAGE = BASE / "data/turtle_stage1_step024000_smoke_inputs_s1"
MANIFEST = BASE / "outputs/logs/turtle_stage1_step024000_smoke_s1/input_manifest.json"


def images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.name),
    )


def five_ending_at(files: list[Path], target_index: int) -> list[Path]:
    start = min(max(0, target_index - 4), len(files) - 5)
    return files[start : start + 5]


def main() -> None:
    config = json.loads((HERE / "scenes.json").read_text())
    selections = {}
    for scene in ("motion_blurcoffee", "defocus_cisco"):
        cfg = config[scene]
        files = images(Path(cfg["raw_dir"]))
        sharp = json.loads(Path(cfg["sharp_json"]).read_text())
        if not sharp:
            raise RuntimeError(f"{scene}: no frozen sharp frame for smoke target")
        by_stem = {path.stem: index for index, path in enumerate(files)}
        target = next((name for name in sharp if Path(name).stem in by_stem), None)
        if target is None:
            raise RuntimeError(f"{scene}: sharp manifest does not match raw sequence")
        window = five_ending_at(files, by_stem[Path(target).stem])
        selections[scene] = {"files": window, "target": window[-1].name, "authoritative_sharp": True}

    tum = config["tum_fr2_xyz"]
    tum_files = images(Path(tum["raw_dir"]))
    evaluation_names = json.loads(Path(tum["evaluation_manifest"]["path"]).read_text())
    # Pick a non-boundary authoritative mapping keyframe so four genuine
    # preceding raw frames provide ordered history.
    target_index = next(int(Path(name).stem) for name in evaluation_names if int(Path(name).stem) >= 4)
    window = five_ending_at(tum_files, target_index)
    selections["tum_fr2_xyz"] = {
        "files": window,
        "target": window[-1].name,
        "authoritative_sharp": True,
        "evaluation_mapping_name": f"{target_index:06d}",
    }

    rows = {}
    for scene, selection in selections.items():
        destination = STAGE / scene
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise RuntimeError(f"refusing populated smoke input: {destination}")
        for index, source in enumerate(selection["files"]):
            (destination / f"{index:06d}{source.suffix.lower()}").symlink_to(source.resolve())
        rows[scene] = {
            "ordered_sources": [str(path.resolve()) for path in selection["files"]],
            "target_source": str(selection["files"][-1].resolve()),
            "authoritative_sharp": selection["authoritative_sharp"],
            **({"evaluation_mapping_name": selection["evaluation_mapping_name"]} if "evaluation_mapping_name" in selection else {}),
        }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
