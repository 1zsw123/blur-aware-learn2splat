#!/usr/bin/env python3
"""Fail closed when an ExBlur hold render reproduces the old ~19 dB bug."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from PIL import Image


SCENES = (
    "exblur_bench",
    "exblur_camellia",
    "exblur_dragon",
    "exblur_jars",
    "exblur_jars2",
    "exblur_postbox",
    "exblur_stone_lantern",
    "exblur_sunflowers",
)
STEPS = (10000, 20000, 30000, 40000, 50000)
ROW_COUNTS = {scene: (3 if scene == "exblur_dragon" else 4) for scene in SCENES}


def grid_psnr(path: Path, rows: int) -> tuple[float, float]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    height, width = image.shape[:2]
    if width % 3 or height % rows:
        raise ValueError(f"unexpected hold grid dimensions {width}x{height}")
    view_width = width // 3
    view_height = height // rows - 24
    if view_height <= 0:
        raise ValueError(f"invalid inferred view height {view_height}")
    psnrs = []
    black_fractions = []
    for row in range(rows):
        y0 = row * (view_height + 24)
        target = image[y0 : y0 + view_height, view_width : 2 * view_width]
        render = image[y0 : y0 + view_height, 2 * view_width : 3 * view_width]
        mse = float(np.square(target - render).mean())
        psnrs.append(120.0 if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse))
        black_fractions.append(float((render.mean(axis=2) < 3.0).mean()))
    return float(np.mean(psnrs)), float(np.mean(black_fractions))


def scene_pids(output_root: Path, scene: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-af", "run_cross_dataset.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    matches = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.partition(" ")
        if str(output_root) in command and f"--scene {scene} " in command:
            matches.append(int(pid_text))
    return matches


def append_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys(), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-status", type=Path, required=True)
    parser.add_argument("--gate-log", type=Path, required=True)
    parser.add_argument("--minimum-psnr", type=float, default=25.0)
    parser.add_argument("--maximum-black-fraction", type=float, default=0.25)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    checked: set[tuple[str, int]] = set()
    previous_psnr: dict[str, float] = {}
    while len(checked) < len(SCENES) * len(STEPS):
        for scene in SCENES:
            for step in STEPS:
                key = (scene, step)
                if key in checked:
                    continue
                grid = (
                    args.output_root
                    / scene
                    / "blur-aware"
                    / f"hold_step_{step}.png"
                )
                if not grid.is_file() or grid.stat().st_size == 0:
                    continue
                psnr, black_fraction = grid_psnr(grid, ROW_COUNTS[scene])
                passed = (
                    psnr >= args.minimum_psnr
                    and black_fraction <= args.maximum_black_fraction
                )
                delta = (
                    None
                    if scene not in previous_psnr
                    else psnr - previous_psnr[scene]
                )
                status = "PASS" if passed else "FAIL_STOP"
                if passed and delta is not None and delta < -1.5:
                    status = "PASS_REGRESSION_WARNING"
                pids = scene_pids(args.output_root, scene)
                append_row(
                    args.gate_log,
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "scene": scene,
                        "step": step,
                        "quantized_grid_psnr": f"{psnr:.6f}",
                        "delta_from_previous": (
                            "" if delta is None else f"{delta:.6f}"
                        ),
                        "render_black_fraction": f"{black_fraction:.8f}",
                        "minimum_psnr": args.minimum_psnr,
                        "status": status,
                        "active_pids": ",".join(map(str, pids)),
                    },
                )
                checked.add(key)
                previous_psnr[scene] = psnr
                if not passed:
                    for pid in pids:
                        os.kill(pid, signal.SIGTERM)

        if args.queue_status.exists() and "QUEUE_TERMINAL" in args.queue_status.read_text():
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
