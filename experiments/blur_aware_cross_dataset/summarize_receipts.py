#!/usr/bin/env python3
"""Create reproducible tables and curves from cross-dataset receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each immutable receipt to compare.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_runs(specs: list[str]) -> list[dict]:
    runs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"receipt must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        receipt = json.loads(path.read_text())
        metrics = receipt.get("metrics", [])
        if not metrics:
            raise RuntimeError(f"receipt has no metrics: {path}")
        runs.append(
            {
                "label": label,
                "path": path,
                "scene": receipt["scene"],
                "contract": receipt["same_config_contract"],
                "metrics": metrics,
                "capacity_events": receipt.get("capacity_events", []),
            }
        )
    return runs


def write_csv(path: Path, runs: list[dict]) -> None:
    fieldnames = (
        "label",
        "scene",
        "adc",
        "step",
        "progress",
        "psnr",
        "ssim",
        "lpips",
        "num_gaussians",
        "receipt",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            final_step = max(int(metric["step"]) for metric in run["metrics"])
            for metric in run["metrics"]:
                writer.writerow(
                    {
                        "label": run["label"],
                        "scene": run["scene"],
                        "adc": run["contract"]["adc"],
                        "step": metric["step"],
                        "progress": float(metric["step"]) / final_step,
                        "psnr": metric["hold_psnr"],
                        "ssim": metric["hold_ssim"],
                        "lpips": metric.get("hold_lpips"),
                        "num_gaussians": metric["num_gaussians"],
                        "receipt": run["path"],
                    }
                )


def write_final_csv(path: Path, runs: list[dict]) -> None:
    """Write one audit-friendly best/final row per immutable run."""
    fieldnames = (
        "label",
        "scene",
        "adc",
        "controller_version",
        "best_step",
        "best_psnr",
        "best_ssim_at_best_psnr",
        "best_lpips_at_best_psnr",
        "final_step",
        "final_psnr",
        "final_ssim",
        "final_lpips",
        "final_num_gaussians",
        "receipt",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            metrics = sorted(run["metrics"], key=lambda item: int(item["step"]))
            best = max(metrics, key=lambda item: float(item["hold_psnr"]))
            final = metrics[-1]
            controller = run["contract"].get("capacity_controller") or {}
            writer.writerow(
                {
                    "label": run["label"],
                    "scene": run["scene"],
                    "adc": run["contract"]["adc"],
                    "controller_version": controller.get("version"),
                    "best_step": best["step"],
                    "best_psnr": best["hold_psnr"],
                    "best_ssim_at_best_psnr": best["hold_ssim"],
                    "best_lpips_at_best_psnr": best.get("hold_lpips"),
                    "final_step": final["step"],
                    "final_psnr": final["hold_psnr"],
                    "final_ssim": final["hold_ssim"],
                    "final_lpips": final.get("hold_lpips"),
                    "final_num_gaussians": final["num_gaussians"],
                    "receipt": run["path"],
                }
            )


def write_per_view_csv(path: Path, runs: list[dict]) -> None:
    """Preserve final per-view distributions so averages cannot hide failures."""
    fieldnames = (
        "label",
        "scene",
        "view_index",
        "final_step",
        "psnr",
        "lpips",
        "receipt",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            final = max(run["metrics"], key=lambda item: int(item["step"]))
            psnr = final.get("hold_per_view_psnr") or []
            lpips = final.get("hold_per_view_lpips")
            if lpips is not None and len(lpips) != len(psnr):
                raise RuntimeError(
                    f"per-view PSNR/LPIPS length mismatch in {run['path']}"
                )
            for index, value in enumerate(psnr):
                writer.writerow(
                    {
                        "label": run["label"],
                        "scene": run["scene"],
                        "view_index": index,
                        "final_step": final["step"],
                        "psnr": value,
                        "lpips": None if lpips is None else lpips[index],
                        "receipt": run["path"],
                    }
                )


def write_capacity_csv(path: Path, runs: list[dict]) -> None:
    """Export the controller state at every structural decision."""
    fieldnames = (
        "label",
        "scene",
        "controller_version",
        "step",
        "support_fraction",
        "rewarded_support_fraction",
        "capacity_pressure",
        "growth_fraction",
        "visible_fraction",
        "base_cap",
        "demand_multiplier",
        "effective_cap",
        "growth_budget",
        "reward_feedback_revision",
        "reward_used",
        "probe_psnr_delta",
        "probe_surplus_delta",
        "quality_reward",
        "complexity_cost",
        "densification_reward",
        "densification_reward_ema",
        "action_factor",
        "cloned",
        "split",
        "pruned",
        "num_gaussians",
        "receipt",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            controller = run["contract"].get("capacity_controller") or {}
            for event in run["capacity_events"]:
                writer.writerow(
                    {
                        "label": run["label"],
                        "scene": run["scene"],
                        "controller_version": controller.get("version"),
                        **{key: event.get(key) for key in fieldnames[3:-1]},
                        "receipt": run["path"],
                    }
                )


def write_plot(path: Path, runs: list[dict]) -> None:
    metrics = (
        ("hold_psnr", "PSNR (dB)"),
        ("hold_ssim", "SSIM"),
        ("hold_lpips", "LPIPS"),
        ("num_gaussians", "Primitives"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (key, ylabel) in zip(axes.flat, metrics):
        for run in runs:
            points = [metric for metric in run["metrics"] if metric.get(key) is not None]
            if not points:
                continue
            final_step = max(int(metric["step"]) for metric in run["metrics"])
            x = [float(metric["step"]) / final_step for metric in points]
            y = [float(metric[key]) for metric in points]
            axis.plot(x, y, marker="o", linewidth=2, label=run["label"])
        axis.set_xlabel("Normalized optimization progress")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=min(4, len(labels)))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    runs = load_runs(args.receipt)
    write_csv(output_dir / "metrics.csv", runs)
    write_final_csv(output_dir / "final_metrics.csv", runs)
    write_per_view_csv(output_dir / "per_view_final_metrics.csv", runs)
    write_capacity_csv(output_dir / "capacity_events.csv", runs)
    write_plot(output_dir / "metrics_curves.png", runs)


if __name__ == "__main__":
    main()
