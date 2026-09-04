#!/usr/bin/env python3
"""Build same-view PRISM3D teacher and checkpoint diagnostic montages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_ROOT = Path("/srv2/szha0669/blur_slam_exp/data")
OUTPUTS_ROOT = Path("/srv2/szha0669/blur_slam_exp/outputs")
RUN_10K = OUTPUTS_ROOT / (
    "prism3d_e8_turtle_step024000_dense25_nima06_identityblind_"
    "allframes_10k_s1"
)
RUN_50K = OUTPUTS_ROOT / (
    "prism3d_e4_turtle_step024000_dense25_nima06_identityblind_"
    "allframes_50k_s1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("jars", "jars2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.mean(
        (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    )
    return 100.0 if error == 0.0 else 10.0 * math.log10(255.0**2 / error)


def extract_outputs(scene: str, run_root: Path, filename: str) -> dict[str, np.ndarray]:
    base = run_root / f"prism3d_{scene}_turtle" / "blur-aware"
    rows = json.loads((base / filename.replace(".png", ".json")).read_text())["rows"]
    montage = load_rgb(base / filename)
    result = {}
    for row_index, row in enumerate(rows):
        top = row_index * 570
        result[row["name"]] = montage[top : top + 540, 800:1600].copy()
    return result


def label_tile(image: np.ndarray, title: str, subtitle: str, width: int = 400) -> Image.Image:
    tile = Image.fromarray(image).resize((width, 270), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, 320), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=14)
    draw.text((8, 276), title, fill="black", font=font)
    draw.text((8, 298), subtitle, fill=(70, 70, 70), font=small)
    return canvas


def worst_regression_crop(
    ours_10k: np.ndarray,
    ours_50k: np.ndarray,
    sharp: np.ndarray,
    crop_w: int = 260,
    crop_h: int = 180,
) -> tuple[int, int, int, int, float]:
    error_10k = np.mean(
        (ours_10k.astype(np.float32) - sharp.astype(np.float32)) ** 2, axis=2
    )
    error_50k = np.mean(
        (ours_50k.astype(np.float32) - sharp.astype(np.float32)) ** 2, axis=2
    )
    regression = error_50k - error_10k
    integral = np.pad(regression, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = regression.shape
    best = (-float("inf"), 0, 0)
    for y in range(0, h - crop_h + 1, 12):
        y2 = y + crop_h
        for x in range(0, w - crop_w + 1, 12):
            x2 = x + crop_w
            score = (
                integral[y2, x2]
                - integral[y, x2]
                - integral[y2, x]
                + integral[y, x]
            ) / (crop_w * crop_h)
            if score > best[0]:
                best = (float(score), x, y)
    _, x, y = best
    return x, y, x + crop_w, y + crop_h, best[0]


def main() -> None:
    args = parse_args()
    scene = args.scene
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_10k = extract_outputs(scene, RUN_10K, "blurred_input_vs_output_top3.png")
    outputs_50k = extract_outputs(
        scene, RUN_50K, "blurred_input_vs_output_selected_50k.png"
    )
    names = list(outputs_10k)
    if names != list(outputs_50k):
        raise RuntimeError("10K and 50K montages do not contain the same views")

    scene_root = DATA_ROOT / "exblurnerf/exblur_release" / scene
    turtle_root = DATA_ROOT / "turtle_stage1_step024000_exblur8_s1" / f"exblur_{scene}"
    columns = ("RAW input", "Turtle step24K", "Ours 10K", "Ours 50K", "Sharp GT")
    rows = []
    zoom_rows = []
    report = {"scene": scene, "rows": []}
    for name in names:
        raw = load_rgb(scene_root / "images" / f"{name}.png")
        turtle = load_rgb(turtle_root / f"{name}.png")
        ours_10k = outputs_10k[name]
        ours_50k = outputs_50k[name]
        sharp = load_rgb(scene_root / "images_sharp" / f"{name}.png")
        images = (raw, turtle, ours_10k, ours_50k, sharp)
        values = [psnr(image, sharp) for image in images]
        rows.append(
            [
                label_tile(image, title, f"frame {name} | PSNR {value:.2f} dB")
                for image, title, value in zip(images, columns, values)
            ]
        )

        x1, y1, x2, y2, regression = worst_regression_crop(ours_10k, ours_50k, sharp)
        zooms = tuple(image[y1:y2, x1:x2] for image in images)
        zoom_values = [psnr(image, zooms[-1]) for image in zooms]
        zoom_rows.append(
            [
                label_tile(image, title, f"frame {name} crop | PSNR {value:.2f} dB")
                for image, title, value in zip(zooms, columns, zoom_values)
            ]
        )
        report["rows"].append(
            {
                "name": name,
                "full_psnr": dict(zip(columns, values)),
                "regression_crop_xyxy": [x1, y1, x2, y2],
                "regression_mse_delta_50k_minus_10k": regression,
                "crop_psnr": dict(zip(columns, zoom_values)),
            }
        )

    def save_grid(grid: list[list[Image.Image]], suffix: str) -> None:
        cell_w, cell_h = grid[0][0].size
        canvas = Image.new("RGB", (cell_w * len(columns), cell_h * len(grid)), "white")
        for row_index, row in enumerate(grid):
            for col_index, tile in enumerate(row):
                canvas.paste(tile, (col_index * cell_w, row_index * cell_h))
        canvas.save(args.output_dir / f"{scene}_{suffix}.png")

    save_grid(rows, "same_view_10k_50k_teacher_gt")
    save_grid(zoom_rows, "worst_50k_regression_crops")
    (args.output_dir / f"{scene}_same_view_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
