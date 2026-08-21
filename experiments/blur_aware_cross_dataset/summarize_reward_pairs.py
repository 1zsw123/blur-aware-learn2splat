#!/usr/bin/env python3
"""Summarize matched densification-reward/control receipt pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=4,
        required=True,
        metavar=("DOMAIN", "SEED", "REWARD_RECEIPT", "CONTROL_RECEIPT"),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_metric(receipt: dict) -> dict:
    metrics = receipt.get("metrics") or []
    if not metrics:
        raise RuntimeError("receipt has no metrics")
    return max(metrics, key=lambda item: int(item["step"]))


def comparable_contract(receipt: dict) -> dict:
    contract = dict(receipt["same_config_contract"])
    contract.pop("densification_reward", None)
    return contract


def load_pair(spec: list[str]) -> dict:
    domain, seed_text, reward_text, control_text = spec
    seed = int(seed_text)
    reward_path = Path(reward_text)
    control_path = Path(control_text)
    reward = json.loads(reward_path.read_text())
    control = json.loads(control_path.read_text())
    if reward["scene"] != control["scene"]:
        raise RuntimeError(f"scene mismatch for {domain}/{seed}")
    if reward["same_config_contract"]["seed"] != seed:
        raise RuntimeError(f"reward seed mismatch for {domain}/{seed}")
    if control["same_config_contract"]["seed"] != seed:
        raise RuntimeError(f"control seed mismatch for {domain}/{seed}")
    if reward["same_config_contract"]["densification_reward"] != "surplus_probe":
        raise RuntimeError(f"reward mode mismatch for {domain}/{seed}")
    if control["same_config_contract"]["densification_reward"] != "probe_control":
        raise RuntimeError(f"control mode mismatch for {domain}/{seed}")
    if comparable_contract(reward) != comparable_contract(control):
        raise RuntimeError(f"non-reward contract mismatch for {domain}/{seed}")

    reward_metric = final_metric(reward)
    control_metric = final_metric(control)
    action_events = [
        event for event in reward.get("capacity_events", []) if event.get("reward_used")
    ]
    if not action_events:
        raise RuntimeError(f"no consumed action reward for {domain}/{seed}")
    if not any(
        abs(float(event.get("probe_surplus_delta", 0.0))) > 0.0
        for event in action_events
    ):
        raise RuntimeError(f"surplus never entered reward for {domain}/{seed}")

    return {
        "domain": domain,
        "scene": reward["scene"],
        "seed": seed,
        "step": int(reward_metric["step"]),
        "reward_psnr": float(reward_metric["hold_psnr"]),
        "control_psnr": float(control_metric["hold_psnr"]),
        "delta_psnr": float(reward_metric["hold_psnr"])
        - float(control_metric["hold_psnr"]),
        "reward_ssim": float(reward_metric["hold_ssim"]),
        "control_ssim": float(control_metric["hold_ssim"]),
        "delta_ssim": float(reward_metric["hold_ssim"])
        - float(control_metric["hold_ssim"]),
        "reward_lpips": float(reward_metric["hold_lpips"]),
        "control_lpips": float(control_metric["hold_lpips"]),
        "delta_lpips": float(reward_metric["hold_lpips"])
        - float(control_metric["hold_lpips"]),
        "reward_primitives": int(reward_metric["num_gaussians"]),
        "control_primitives": int(control_metric["num_gaussians"]),
        "delta_primitives": int(reward_metric["num_gaussians"])
        - int(control_metric["num_gaussians"]),
        "mean_action_factor": float(
            np.mean([float(event["action_factor"]) for event in action_events])
        ),
        "surplus_reward_events": len(action_events),
        "reward_receipt": str(reward_path),
        "reward_receipt_sha256": sha256(reward_path),
        "control_receipt": str(control_path),
        "control_receipt_sha256": sha256(control_path),
    }


def means(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    result = []
    for domain, items in grouped.items():
        result.append(
            {
                "domain": domain,
                "seeds": [item["seed"] for item in items],
                "delta_psnr": float(np.mean([item["delta_psnr"] for item in items])),
                "delta_ssim": float(np.mean([item["delta_ssim"] for item in items])),
                "delta_lpips": float(np.mean([item["delta_lpips"] for item in items])),
                "delta_primitives": float(
                    np.mean([item["delta_primitives"] for item in items])
                ),
                "mean_action_factor": float(
                    np.mean([item["mean_action_factor"] for item in items])
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict], mean_rows: list[dict]) -> None:
    domains = [row["domain"] for row in mean_rows]
    colors = dict(zip(domains, ("#2176AE", "#2E8B57", "#D97706")))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("delta_psnr", "Reward - control PSNR (dB)", 1.0),
        ("delta_ssim", "Reward - control SSIM", 1.0),
        ("delta_lpips", "LPIPS improvement (control - reward)", -1.0),
        ("delta_primitives", "Reward - control primitives", 1.0),
    )
    x = np.arange(len(domains))
    for axis, (key, label, sign) in zip(axes.flat, panels):
        values = [sign * row[key] for row in mean_rows]
        axis.bar(x, values, color=[colors[domain] for domain in domains], alpha=0.85)
        for domain_index, domain in enumerate(domains):
            samples = [
                sign * row[key] for row in rows if row["domain"] == domain
            ]
            offsets = np.linspace(-0.08, 0.08, len(samples))
            axis.scatter(
                domain_index + offsets,
                samples,
                color="black",
                marker="o",
                s=28,
                zorder=3,
            )
        axis.axhline(0.0, color="#333333", linewidth=1)
        axis.set_xticks(x, domains)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Surplus-conditioned densification reward vs fixed-probe control\n"
        "10K smoke, bars = two-seed mean, dots = individual seeds",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [load_pair(spec) for spec in args.pair]
    mean_rows = means(rows)
    write_csv(output_dir / "paired_seed_results.csv", rows)
    write_csv(output_dir / "domain_means.csv", mean_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"runs": rows, "domain_means": mean_rows}, indent=2) + "\n"
    )
    plot_summary(output_dir / "reward_vs_probe_control.png", rows, mean_rows)


if __name__ == "__main__":
    main()
