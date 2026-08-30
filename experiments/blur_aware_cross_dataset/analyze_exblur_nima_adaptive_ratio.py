#!/usr/bin/env python3
"""Compare hold-blind NIMA mixture selection with ExBlur hold ratios."""

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


def fit_scene(score_path: Path, hold_path: Path) -> dict[str, object]:
    rows = json.loads(score_path.read_text())
    discovery = discover_sharp_anchors(rows)
    selected = set(discovery["selected_frames"])

    # Hold identities are loaded only after fitting and are used only for reporting.
    holds = {Path(name).stem for name in json.loads(hold_path.read_text())}
    overlap = selected & holds

    return {
        "frame_count": len(rows),
        "hold_count": len(holds),
        "hold_percent": 100.0 * len(holds) / len(rows),
        "automatic_count": len(selected),
        "automatic_percent": 100.0 * len(selected) / len(rows),
        "selected_frames": sorted(selected),
        "hold_frames": sorted(holds),
        "overlap_count": len(overlap),
        "precision_vs_hold": len(overlap) / len(selected),
        "recall_vs_hold": len(overlap) / len(holds),
        "component_separation": discovery["component_separation"],
        "bic_gain_two_vs_one": discovery["bic_gain_two_vs_one"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        scene: fit_scene(
            args.logs_root / f"{scene}_exblur_nima_koniq_scores.json",
            args.logs_root
            / "exblur8_official_protocol_s1"
            / f"{scene}_official_test.json",
        )
        for scene in SCENES
    }
    total_frames = sum(int(row["frame_count"]) for row in report.values())
    total_holds = sum(int(row["hold_count"]) for row in report.values())
    total_selected = sum(int(row["automatic_count"]) for row in report.values())
    total_overlap = sum(int(row["overlap_count"]) for row in report.values())
    report["aggregate"] = {
        "frame_count": total_frames,
        "hold_count": total_holds,
        "hold_percent": 100.0 * total_holds / total_frames,
        "automatic_count": total_selected,
        "automatic_percent": 100.0 * total_selected / total_frames,
        "overlap_count": total_overlap,
        "precision_vs_hold": total_overlap / total_selected,
        "recall_vs_hold": total_overlap / total_holds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
