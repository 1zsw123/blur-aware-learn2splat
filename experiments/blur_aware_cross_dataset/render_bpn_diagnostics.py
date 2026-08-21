#!/usr/bin/env python3
"""Regenerate stratified BPN diagnostics from a completed run receipt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments.blur_aware_cross_dataset.run_cross_dataset import (
    save_kernel_visualization,
)
from optgs.experimental.blur_aware import BlurAwareObjective, BlurAwareObjectiveConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    payload = torch.load(
        run_dir / "blur_aware_objective.pt", map_location="cpu", weights_only=False
    )
    model_state = payload["model"]
    num_views = int(model_state["bpn.camera_embedding.weight"].shape[0])
    objective = BlurAwareObjective(
        num_views, BlurAwareObjectiveConfig(**payload["config"])
    )
    objective.load_state_dict(model_state)
    objective.eval()

    receipt = json.loads((run_dir / "receipt.json").read_text())
    with (run_dir / "reliability.csv").open(newline="") as stream:
        reliability = list(csv.DictReader(stream))
    optimization_indices = [int(value) for value in receipt["optimization_indices"]]
    selected = [reliability[index] for index in optimization_indices]
    if len(selected) != num_views:
        raise RuntimeError(
            f"receipt has {len(selected)} optimization views but BPN has {num_views}"
        )
    names = [row["name"] for row in selected]
    known_sharp = torch.tensor(
        [row["known_sharp"].lower() == "true" for row in selected],
        dtype=torch.bool,
    )
    save_kernel_visualization(
        run_dir / "bpn_kernels_stratified.png",
        objective,
        names,
        known_sharp,
        run_dir / "bpn_kernel_stats.csv",
    )
    print(run_dir / "bpn_kernels_stratified.png")
    print(run_dir / "bpn_kernel_stats.csv")


if __name__ == "__main__":
    main()
