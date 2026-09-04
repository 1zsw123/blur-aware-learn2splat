#!/usr/bin/env python3
"""Prepare hold-blind top-percent NIMA supervision for ExBlur sunflowers."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_CONFIG = HERE / "scenes_exblur8_evssm.generated.json"
SCORES = Path(
    "/srv2/szha0669/blur_slam_exp/outputs/logs/"
    "sunflowers_exblur_nima_koniq_scores.json"
)
OUTPUT_DIR = Path(
    "/srv2/szha0669/blur_slam_exp/outputs/logs/"
    "exblur_sunflowers_nima_quantile_ablation_s1"
)
PERCENTS = (25, 20, 15, 10)


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text())["exblur_sunflowers"]
    rows = sorted(
        json.loads(SCORES.read_text()),
        key=lambda row: (-float(row["nima_koniq"]), str(row["name"])),
    )
    if len(rows) != 43 or len({str(row["name"]) for row in rows}) != len(rows):
        raise RuntimeError("sunflowers NIMA score manifest is incomplete or duplicated")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    configs = {}
    summary = []
    for percent in PERCENTS:
        count = math.ceil(len(rows) * percent / 100.0)
        selected = rows[:count]
        sharp_path = OUTPUT_DIR / f"top{percent}_sharp_frames.json"
        sharp_path.write_text(
            json.dumps([str(row["name"]) for row in selected], indent=2) + "\n"
        )
        scene = f"exblur_sunflowers_nimatop{percent}"
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
        configs[scene] = cfg
        summary.append(
            {
                "scene": scene,
                "top_percent": percent,
                "selected_count": count,
                "threshold_score": float(selected[-1]["nima_koniq"]),
                "selected": selected,
            }
        )

    (OUTPUT_DIR / "scenes.json").write_text(json.dumps(configs, indent=2) + "\n")
    (OUTPUT_DIR / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
