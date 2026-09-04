#!/usr/bin/env python3
"""Low-memory longitudinal diagnostics for completed PRISM3D runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DIAGNOSTIC_COLUMNS = (
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
CAPACITY_COLUMNS = (
    "valid",
    "cloned",
    "split",
    "pruned",
    "cap_truncated",
    "reward_mean",
    "blur_quality_reward",
    "blur_psnr_delta",
    "blur_raw_psnr_delta",
    "blur_surplus_delta",
    "blur_structural_fraction",
    "blur_capacity_cost",
    "blur_action_support_mean",
    "blur_birth_penalty_gate_mean",
    "blur_net_action_direction",
    "num_gaussians",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def aggregate_csv(path: Path) -> list[dict]:
    sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, int] = defaultdict(int)
    lasts: dict[int, dict[str, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["step"]))
            window = step // 10000
            counts[window] += 1
            current = {}
            for column in DIAGNOSTIC_COLUMNS:
                value = float(row[column])
                sums[window][column] += value
                current[column] = value
            current["step"] = step
            lasts[window] = current
    output = []
    for window in sorted(counts):
        record = {
            "start": window * 10000,
            "end": (window + 1) * 10000,
            "count": counts[window],
            "last_step": lasts[window]["step"],
        }
        for column in DIAGNOSTIC_COLUMNS:
            record[f"{column}_mean"] = sums[window][column] / counts[window]
            record[f"{column}_last"] = lasts[window][column]
        output.append(record)
    return output


def aggregate_capacity(events: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        groups[int(event["step"]) // 10000].append(event)
    output = []
    for window, rows in sorted(groups.items()):
        record = {
            "start": window * 10000,
            "end": (window + 1) * 10000,
            "events": len(rows),
            "first_step": int(rows[0]["step"]),
            "last_step": int(rows[-1]["step"]),
        }
        for column in CAPACITY_COLUMNS:
            values = [float(row[column]) for row in rows if row.get(column) is not None]
            if values:
                record[f"{column}_mean"] = float(np.mean(values))
                record[f"{column}_last"] = values[-1]
                record[f"{column}_sum"] = float(np.sum(values))
        output.append(record)
    return output


def bpn_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    result = {"views": len(rows)}
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
        values = np.asarray([float(row[column]) for row in rows])
        result[f"{column}_mean"] = float(values.mean())
        result[f"{column}_median"] = float(np.median(values))
        result[f"{column}_p90"] = float(np.quantile(values, 0.9))
    radius = np.asarray([float(row["rms_radius_px"]) for row in rows])
    result["radius_gt8_fraction"] = float(np.mean(radius > 8.0))
    result["radius_lt1_fraction"] = float(np.mean(radius < 1.0))
    return result


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    receipt = json.loads((run_dir / "receipt.json").read_text())
    report = {
        "scene": receipt["scene"],
        "metrics": receipt["metrics"],
        "diagnostic_windows": aggregate_csv(run_dir / "training_diagnostics.csv"),
        "capacity_windows": aggregate_capacity(receipt["capacity_events"]),
        "bpn": {
            str(step): bpn_metrics(run_dir / f"bpn_kernel_stats_step_{step}.csv")
            for step in (10000, 20000, 30000, 40000, 50000)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
