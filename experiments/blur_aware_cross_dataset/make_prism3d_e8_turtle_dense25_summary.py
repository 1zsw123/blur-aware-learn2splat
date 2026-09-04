#!/usr/bin/env python3
"""Summarize the frozen PRISM3D-E Turtle dense-25 experiment."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(
    "/srv2/szha0669/blur_slam_exp/outputs/"
    "prism3d_e8_turtle_step024000_dense25_nima06_identityblind_allframes_10k_s1"
)
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


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    rows = []
    for scene in SCENES:
        scene_id = f"prism3d_{scene}_turtle"
        receipt_path = ROOT / scene_id / "blur-aware/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        metric = receipt["metrics"][-1]
        rows.append(
            {
                "scene": scene,
                "psnr": metric["hold_psnr"],
                "ssim": metric["hold_ssim"],
                "lpips": metric["hold_lpips"],
                "primitives": metric["num_gaussians"],
                "optimization_views": len(receipt["optimization_indices"]),
                "evaluation_views": len(receipt["evaluation_indices"]),
                "nima06_sharp_anchors": receipt["sharp_supervision"]["count"],
            }
        )

    csv_path = ROOT / "prism3d_e8_turtle_dense25_nima06_identityblind_10k_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    panel_width, panel_height = 760, 570
    header_height, label_height = 92, 54
    canvas = Image.new(
        "RGB",
        (2 * panel_width, header_height + 4 * (panel_height + label_height)),
        "#101214",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 12),
        "PRISM3D-E synthetic 8 scenes | Turtle step-24K | NIMA>0.6 | dense 25x25 | 10K",
        fill="white",
        font=font(23),
    )
    draw.text(
        (20, 48),
        "All frames optimized; hold identity hidden from supervision; left: blurred input, right: Ours",
        fill="#b8c0c8",
        font=font(18),
    )
    for slot, (scene, row) in enumerate(zip(SCENES, rows)):
        scene_id = f"prism3d_{scene}_turtle"
        source = ROOT / scene_id / "blur-aware/blurred_input_vs_output_top3.png"
        image = Image.open(source).convert("RGB")
        fitted = ImageOps.contain(image, (panel_width - 20, panel_height - 10), Image.Resampling.LANCZOS)
        column, grid_row = slot % 2, slot // 2
        x = column * panel_width + (panel_width - fitted.width) // 2
        y = header_height + grid_row * (panel_height + label_height)
        canvas.paste(fitted, (x, y))
        label = (
            f"{scene}  PSNR {row['psnr']:.2f}  SSIM {row['ssim']:.4f}  "
            f"LPIPS {row['lpips']:.4f}"
        )
        draw.text((column * panel_width + 18, y + panel_height + 8), label, fill="#efb366", font=font(18))

    montage_path = ROOT / "prism3d_e8_blurred_input_vs_output_top3_montage.png"
    canvas.save(montage_path)

    averages = {key: sum(row[key] for row in rows) / len(rows) for key in ("psnr", "ssim", "lpips")}
    print(csv_path)
    print(montage_path)
    print(json.dumps(averages, sort_keys=True))


if __name__ == "__main__":
    main()
