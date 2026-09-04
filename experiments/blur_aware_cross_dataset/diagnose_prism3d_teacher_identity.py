#!/usr/bin/env python3
"""Diagnose Turtle corruption and confidence inversion on already-sharp inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
DATA_ROOT = Path("/srv2/szha0669/blur_slam_exp/data")
RUN_ROOT = Path(
    "/srv2/szha0669/blur_slam_exp/outputs/"
    "prism3d_e8_turtle_step024000_dense25_nima06_identityblind_"
    "allframes_10k_s1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


def tile(image: np.ndarray, title: str, subtitle: str) -> Image.Image:
    width, height = 400, 270
    displayed = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 50), "white")
    canvas.paste(displayed)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, height + 5), title, fill="black", font=ImageFont.load_default(size=16))
    draw.text(
        (8, height + 27),
        subtitle,
        fill=(70, 70, 70),
        font=ImageFont.load_default(size=14),
    )
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"definition": "sharp input is RAW exactly equal to sharp GT", "scenes": {}}
    grid = []
    for scene in SCENES:
        scene_root = DATA_ROOT / "exblurnerf/exblur_release" / scene
        turtle_root = (
            DATA_ROOT / "turtle_stage1_step024000_exblur8_s1" / f"exblur_{scene}"
        )
        reliability_path = (
            RUN_ROOT / f"prism3d_{scene}_turtle/blur-aware/reliability.csv"
        )
        reliability = {
            row["name"]: float(row["confidence"])
            for row in csv.DictReader(reliability_path.open())
        }
        rows = []
        for raw_path in sorted((scene_root / "images").glob("*.png")):
            raw = load(raw_path)
            sharp = load(scene_root / "images_sharp" / raw_path.name)
            turtle = load(turtle_root / raw_path.name)
            raw_psnr = psnr(raw, sharp)
            rows.append(
                {
                    "name": raw_path.stem,
                    "raw_psnr": raw_psnr,
                    "turtle_psnr": psnr(turtle, sharp),
                    "confidence": reliability[raw_path.stem],
                    "raw": raw,
                    "turtle": turtle,
                    "sharp": sharp,
                }
            )
        sharp_rows = [row for row in rows if row["raw_psnr"] > 60.0]
        blur_rows = [row for row in rows if row["raw_psnr"] <= 60.0]
        if not sharp_rows or not blur_rows:
            raise RuntimeError(f"{scene}: expected both sharp and blurred observations")
        worst = min(sharp_rows, key=lambda row: row["turtle_psnr"])
        absolute_difference = np.abs(
            worst["turtle"].astype(np.int16) - worst["sharp"].astype(np.int16)
        )
        amplified_difference = np.clip(absolute_difference * 4, 0, 255).astype(np.uint8)
        grid.append(
            [
                tile(worst["raw"], f"{scene}: sharp input", f"frame {worst['name']} | RAW=GT"),
                tile(
                    worst["turtle"],
                    "Turtle step24K",
                    f"PSNR {worst['turtle_psnr']:.2f} dB | confidence {worst['confidence']:.3f}",
                ),
                tile(
                    amplified_difference,
                    "|Turtle - sharp| x4",
                    "teacher changed an already-sharp observation",
                ),
            ]
        )
        report["scenes"][scene] = {
            "frames": len(rows),
            "sharp_input_frames": len(sharp_rows),
            "blurred_input_frames": len(blur_rows),
            "confidence_sharp_input_mean": float(
                np.mean([row["confidence"] for row in sharp_rows])
            ),
            "confidence_blurred_input_mean": float(
                np.mean([row["confidence"] for row in blur_rows])
            ),
            "turtle_psnr_on_sharp_input_mean": float(
                np.mean([row["turtle_psnr"] for row in sharp_rows])
            ),
            "raw_psnr_on_blurred_input_mean": float(
                np.mean([row["raw_psnr"] for row in blur_rows])
            ),
            "turtle_psnr_on_blurred_input_mean": float(
                np.mean([row["turtle_psnr"] for row in blur_rows])
            ),
            "worst_sharp_input": {
                "name": worst["name"],
                "turtle_psnr": worst["turtle_psnr"],
                "confidence": worst["confidence"],
            },
        }

    width, height = grid[0][0].size
    canvas = Image.new("RGB", (3 * width, len(grid) * height), "white")
    for row_index, row in enumerate(grid):
        for col_index, image in enumerate(row):
            canvas.paste(image, (col_index * width, row_index * height))
    canvas.save(args.output_dir / "prism3d_teacher_corrupts_sharp_inputs.png")
    (args.output_dir / "prism3d_teacher_identity_diagnosis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
