#!/usr/bin/env python3
"""Prepare a hold-blind NIMA top-percent ablation for one ExBlur scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PERCENTS = (25, 20, 15, 10)
BASE = Path("/srv2/szha0669/blur_slam_exp")
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tag", default="s1")
    args = parser.parse_args()

    configs = json.loads((HERE / "scenes_exblur8_evssm.generated.json").read_text())
    scene_key = f"exblur_{args.scene}"
    if scene_key not in configs:
        raise ValueError(f"unknown ExBlur scene {args.scene!r}")
    base = configs[scene_key]
    score_path = BASE / "outputs/logs" / f"{args.scene}_exblur_nima_koniq_scores.json"
    rows = sorted(
        json.loads(score_path.read_text()),
        key=lambda row: (-float(row["nima_koniq"]), str(row["name"])),
    )
    names = [str(row["name"]) for row in rows]
    image_names = {
        path.stem for path in (Path(base["data_dir"]) / "images").iterdir() if path.is_file()
    }
    if len(set(names)) != len(names) or set(names) != image_names:
        raise RuntimeError("NIMA score names do not exactly match scene input images")

    output_dir = (
        BASE / "outputs/logs" / f"exblur_{args.scene}_nima_quantile_ablation_{args.tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    generated = {}
    summary = []
    for percent in PERCENTS:
        count = math.ceil(len(rows) * percent / 100.0)
        selected = rows[:count]
        sharp_path = output_dir / f"top{percent}_sharp_frames.json"
        sharp_path.write_text(
            json.dumps([str(row["name"]) for row in selected], indent=2) + "\n"
        )
        arm = f"exblur_{args.scene}_nimatop{percent}"
        cfg = dict(base)
        cfg.update(
            {
                "sharp_json": str(sharp_path),
                "sharp_supervision_policy": "sharp_json_only",
                "evaluation_direct_supervision": False,
                "exclude_evaluation_from_optimization": False,
                "require_sharp_evaluation_targets": False,
                "hold_blind_training": True,
                "protocol_description": (
                    f"hold-blind training; scene-relative NIMA top {percent}% "
                    "receives direct RAW supervision, BPN bypass, and w10; "
                    "official hold identity is used only by post-training metrics"
                ),
            }
        )
        generated[arm] = cfg
        summary.append(
            {
                "scene": arm,
                "top_percent": percent,
                "selected_count": count,
                "threshold_score": float(selected[-1]["nima_koniq"]),
                "selected": selected,
            }
        )

    (output_dir / "scenes.json").write_text(json.dumps(generated, indent=2) + "\n")
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
