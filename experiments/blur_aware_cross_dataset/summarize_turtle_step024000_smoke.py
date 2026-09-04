#!/usr/bin/env python3
"""Summarize three-domain Turtle output without claiming final-stage quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left - right) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def laplacian(value: np.ndarray) -> float:
    gray = cv2.cvtColor((value * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, panels = [], []
    for scene, cfg in manifest.items():
        source_paths = [Path(path) for path in cfg["ordered_sources"]]
        prediction_dir = args.prediction_root / "turtle_step024000_smoke" / scene
        predictions = sorted(prediction_dir.glob("Frame_*_Pred.png"), key=lambda path: int(path.name.split("_")[1]))
        if len(predictions) != 5:
            raise RuntimeError(f"{scene}: expected five predictions, got {len(predictions)}")
        for index, (source_path, prediction_path) in enumerate(zip(source_paths, predictions)):
            source, prediction = load(source_path), load(prediction_path)
            if source.shape != prediction.shape:
                raise RuntimeError(f"{scene}/{index}: {source.shape} != {prediction.shape}")
            source_lap, prediction_lap = laplacian(source), laplacian(prediction)
            rows.append({
                "scene": scene,
                "frame": index,
                "source": str(source_path),
                "prediction": str(prediction_path),
                "output_vs_input_psnr": psnr(prediction, source),
                "input_laplacian": source_lap,
                "output_laplacian": prediction_lap,
                "relative_laplacian_change": (prediction_lap - source_lap) / max(source_lap, 1e-6),
                "target_frame": index == 4,
            })
        source = Image.open(source_paths[-1]).convert("RGB")
        prediction = Image.open(predictions[-1]).convert("RGB")
        width = 360
        height = round(source.height * width / source.width)
        source = source.resize((width, height)); prediction = prediction.resize((width, height))
        panel = Image.new("RGB", (width * 2, height + 42), "#111111")
        panel.paste(source, (0, 42)); panel.paste(prediction, (width, 42))
        draw = ImageDraw.Draw(panel)
        target_row = rows[-1]
        draw.text((8, 8), f"{scene}: input", fill="white")
        draw.text((width + 8, 8), f"step24K output  PSNR(input)={target_row['output_vs_input_psnr']:.2f}", fill="white")
        panels.append(panel)
    visual = Image.new("RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels)), "#111111")
    y = 0
    for panel in panels:
        visual.paste(panel, (0, y)); y += panel.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint_stage": "stage1_step24000_nonfinal", "rows": rows}, indent=2) + "\n")
    visual.save(args.visual)
    for scene in manifest:
        values = [row for row in rows if row["scene"] == scene]
        print(scene, "mean_psnr_input", np.mean([row["output_vs_input_psnr"] for row in values]),
              "mean_lap_change", np.mean([row["relative_laplacian_change"] for row in values]))


if __name__ == "__main__":
    main()
