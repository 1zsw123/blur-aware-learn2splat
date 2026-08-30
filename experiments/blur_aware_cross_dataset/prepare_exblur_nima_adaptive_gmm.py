#!/usr/bin/env python3
"""Prepare hold-blind ExBlur configs from adaptive NIMA GMM selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_adaptive_sharp_anchors import discover_sharp_anchors


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configs = json.loads(args.base_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)

    generated = {}
    selection_report = {}
    for scene in SCENES:
        scene_key = f"exblur_{scene}"
        score_path = args.scores_root / f"{scene}_exblur_nima_koniq_scores.json"
        discovery = discover_sharp_anchors(json.loads(score_path.read_text()))
        selected = [str(name) for name in discovery["selected_frames"]]
        sharp_path = args.output_dir / f"{scene}_nima_gmm_sharp_frames.json"
        sharp_path.write_text(json.dumps(selected, indent=2) + "\n")
        selection_report[scene] = {
            "score_path": str(score_path),
            **discovery,
        }

        cfg = dict(configs[scene_key])
        cfg.update(
            {
                "sharp_json": str(sharp_path),
                "sharp_supervision_policy": "sharp_json_only",
                "evaluation_direct_supervision": False,
                "exclude_evaluation_from_optimization": False,
                "require_sharp_evaluation_targets": False,
                "hold_blind_training": True,
                "protocol_description": (
                    "hold-blind training; scene-adaptive two-component NIMA "
                    "mixture selects direct-RAW/BPN-bypass/w10 sharp anchors; "
                    "official hold identity is used only by post-training metrics"
                ),
            }
        )
        generated[scene_key] = cfg

    (args.output_dir / "scenes.json").write_text(
        json.dumps(generated, indent=2) + "\n"
    )
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(selection_report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
