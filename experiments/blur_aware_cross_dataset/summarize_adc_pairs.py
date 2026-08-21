#!/usr/bin/env python3
"""Summarize matched adaptive-controller and exact-LeGS smoke pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("DOMAIN", "ADAPTIVE_RECEIPT", "LEGS_RECEIPT"),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_receipt(path: Path) -> dict:
    receipt = json.loads(path.read_text())
    if not receipt.get("metrics"):
        raise RuntimeError(f"receipt has no metrics: {path}")
    return receipt


def validate_pair(domain: str, adaptive: dict, legs: dict) -> None:
    if adaptive["scene"] != legs["scene"]:
        raise RuntimeError(f"scene mismatch for {domain}")
    if adaptive["same_config_contract"]["adc"] != "adaptive":
        raise RuntimeError(f"adaptive arm has wrong controller for {domain}")
    if legs["same_config_contract"]["adc"] != "legs":
        raise RuntimeError(f"LeGS arm has wrong controller for {domain}")

    contract_keys = (
        "steps_requested",
        "steps_effective",
        "opt_batch_size",
        "opt_batch_strategy_effective",
        "decoder_backend_effective",
        "objective",
        "optimizer",
        "optimizer_switch_step",
        "seed",
    )
    for key in contract_keys:
        left = adaptive["same_config_contract"].get(key)
        right = legs["same_config_contract"].get(key)
        if left != right:
            raise RuntimeError(f"contract mismatch for {domain}: {key}={left!r}/{right!r}")

    receipt_keys = (
        "optimization_indices",
        "evaluation_indices",
        "evaluation_source",
        "evaluation_reference",
        "initialization",
        "camera_preprocess",
        "sharp_supervision",
        "auxiliary_supervision",
        "objective_config",
        "metrics_config",
    )
    for key in receipt_keys:
        if adaptive.get(key) != legs.get(key):
            raise RuntimeError(f"receipt mismatch for {domain}: {key}")

    adaptive_steps = [int(item["step"]) for item in adaptive["metrics"]]
    legs_steps = [int(item["step"]) for item in legs["metrics"]]
    if adaptive_steps != legs_steps:
        raise RuntimeError(f"evaluation-step mismatch for {domain}")


def best_metric(receipt: dict) -> dict:
    return max(receipt["metrics"], key=lambda item: float(item["hold_psnr"]))


def final_metric(receipt: dict) -> dict:
    return max(receipt["metrics"], key=lambda item: int(item["step"]))


def load_pairs(specs: list[list[str]]) -> list[dict]:
    pairs = []
    for domain, adaptive_text, legs_text in specs:
        adaptive_path = Path(adaptive_text)
        legs_path = Path(legs_text)
        adaptive = read_receipt(adaptive_path)
        legs = read_receipt(legs_path)
        validate_pair(domain, adaptive, legs)
        adaptive_best = best_metric(adaptive)
        legs_best = best_metric(legs)
        adaptive_final = final_metric(adaptive)
        legs_final = final_metric(legs)
        pairs.append(
            {
                "domain": domain,
                "scene": adaptive["scene"],
                "adaptive": adaptive,
                "legs": legs,
                "adaptive_path": adaptive_path,
                "legs_path": legs_path,
                "adaptive_sha256": sha256(adaptive_path),
                "legs_sha256": sha256(legs_path),
                "summary": {
                    "adaptive_best_step": int(adaptive_best["step"]),
                    "adaptive_best_psnr": float(adaptive_best["hold_psnr"]),
                    "adaptive_best_ssim": float(adaptive_best["hold_ssim"]),
                    "legs_best_step": int(legs_best["step"]),
                    "legs_best_psnr": float(legs_best["hold_psnr"]),
                    "legs_best_ssim": float(legs_best["hold_ssim"]),
                    "best_psnr_delta": float(legs_best["hold_psnr"])
                    - float(adaptive_best["hold_psnr"]),
                    "best_ssim_delta": float(legs_best["hold_ssim"])
                    - float(adaptive_best["hold_ssim"]),
                    "adaptive_final_psnr": float(adaptive_final["hold_psnr"]),
                    "adaptive_final_ssim": float(adaptive_final["hold_ssim"]),
                    "adaptive_final_primitives": int(adaptive_final["num_gaussians"]),
                    "legs_final_psnr": float(legs_final["hold_psnr"]),
                    "legs_final_ssim": float(legs_final["hold_ssim"]),
                    "legs_final_primitives": int(legs_final["num_gaussians"]),
                    "final_psnr_delta": float(legs_final["hold_psnr"])
                    - float(adaptive_final["hold_psnr"]),
                    "final_ssim_delta": float(legs_final["hold_ssim"])
                    - float(adaptive_final["hold_ssim"]),
                    "final_primitive_ratio": int(legs_final["num_gaussians"])
                    / int(adaptive_final["num_gaussians"]),
                },
            }
        )
    return pairs


def write_summary_csv(path: Path, pairs: list[dict]) -> None:
    rows = [
        {
            "domain": pair["domain"],
            "scene": pair["scene"],
            **pair["summary"],
            "adaptive_receipt": pair["adaptive_path"],
            "adaptive_receipt_sha256": pair["adaptive_sha256"],
            "legs_receipt": pair["legs_path"],
            "legs_receipt_sha256": pair["legs_sha256"],
        }
        for pair in pairs
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_curves_csv(path: Path, pairs: list[dict]) -> None:
    fields = ("domain", "scene", "method", "step", "psnr", "ssim", "primitives")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            for method in ("adaptive", "legs"):
                for metric in pair[method]["metrics"]:
                    writer.writerow(
                        {
                            "domain": pair["domain"],
                            "scene": pair["scene"],
                            "method": method,
                            "step": metric["step"],
                            "psnr": metric["hold_psnr"],
                            "ssim": metric["hold_ssim"],
                            "primitives": metric["num_gaussians"],
                        }
                    )


def write_plot(path: Path, pairs: list[dict]) -> None:
    figure, axes = plt.subplots(
        len(pairs), 3, figsize=(13, 3.6 * len(pairs)), constrained_layout=True
    )
    if len(pairs) == 1:
        axes = [axes]
    columns = (
        ("hold_psnr", "PSNR (dB)"),
        ("hold_ssim", "SSIM"),
        ("num_gaussians", "Primitives"),
    )
    styles = {
        "adaptive": ("#4C78A8", "Adaptive + surplus"),
        "legs": ("#E45756", "Exact LeGS"),
    }
    for row, pair in enumerate(pairs):
        for column, (key, ylabel) in enumerate(columns):
            axis = axes[row][column]
            for method, (color, label) in styles.items():
                metrics = pair[method]["metrics"]
                axis.plot(
                    [int(item["step"]) / 1000 for item in metrics],
                    [float(item[key]) for item in metrics],
                    color=color,
                    marker="o",
                    linewidth=2,
                    label=label,
                )
            axis.set_xlabel("Optimization step (K)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_title(f"{pair['domain']}: {pair['scene']}", loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pairs = load_pairs(args.pair)
    write_summary_csv(output_dir / "paired_summary.csv", pairs)
    write_curves_csv(output_dir / "paired_curves.csv", pairs)
    (output_dir / "paired_summary.json").write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "domain": pair["domain"],
                        "scene": pair["scene"],
                        **pair["summary"],
                        "adaptive_receipt": str(pair["adaptive_path"]),
                        "adaptive_receipt_sha256": pair["adaptive_sha256"],
                        "legs_receipt": str(pair["legs_path"]),
                        "legs_receipt_sha256": pair["legs_sha256"],
                    }
                    for pair in pairs
                ]
            },
            indent=2,
        )
        + "\n"
    )
    write_plot(output_dir / "paired_curves.png", pairs)


if __name__ == "__main__":
    main()
