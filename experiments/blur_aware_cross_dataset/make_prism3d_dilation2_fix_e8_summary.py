#!/usr/bin/env python3
"""Build the eight-scene PRISM3D dilation-2 comparison receipt."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


OUTPUTS = Path("/srv2/szha0669/blur_slam_exp/outputs")
OLD_ROOT = OUTPUTS / "prism3d_e8_turtle_step024000_dense25_nima06_identityblind_allframes_10k_s1"
PAIR_ROOT = OUTPUTS / "prism3d_camellia_stone_kernel25_dilation2_10k_s1"
SIX_ROOT = OUTPUTS / "prism3d_e8_current_dilation2_fix_10k_s1"
RETRY_ROOT = OUTPUTS / "prism3d_bench_jars2_current_dilation2_fix_10k_s2"
SUMMARY_ROOT = OUTPUTS / "prism3d_e8_current_dilation2_fix_10k_summary_s1"
SCENE_ROOTS = {
    "bench": RETRY_ROOT,
    "camellia": PAIR_ROOT,
    "dragon": SIX_ROOT,
    "jars": SIX_ROOT,
    "jars2": RETRY_ROOT,
    "postbox": SIX_ROOT,
    "stone_lantern": PAIR_ROOT,
    "sunflowers": SIX_ROOT,
}


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def load_receipt(root: Path, scene: str) -> tuple[dict, Path]:
    path = root / f"prism3d_{scene}_turtle/blur-aware/receipt.json"
    receipt = json.loads(path.read_text())
    cfg = receipt["objective_config"]
    contract = receipt["same_config_contract"]
    expected = {
        "kernel_size": 25,
        "kernel_dilation": 2,
        "laplacian_loss_mode": "surplus",
        "laplacian_support_mode": "raw_neighborhood",
        "coupled_dual_bpn": True,
        "latent_blur_assignment": True,
        "exposure_trajectory_samples": 1,
    }
    for key, value in expected.items():
        actual = cfg.get(key, 1 if key == "exposure_trajectory_samples" else None)
        if actual != value:
            raise RuntimeError(f"{scene}: {key}={actual!r}, expected {value!r}")
    if contract["steps_effective"] != 10000 or contract["adc"] != "legs_blur":
        raise RuntimeError(f"{scene}: incompatible training contract")
    return receipt, path


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    visuals = []
    for scene, root in SCENE_ROOTS.items():
        current, receipt_path = load_receipt(root, scene)
        old = json.loads(
            (
                OLD_ROOT
                / f"prism3d_{scene}_turtle/blur-aware/receipt.json"
            ).read_text()
        )
        metric = current["metrics"][-1]
        old_metric = old["metrics"][-1]
        rows.append(
            {
                "scene": scene,
                "psnr": metric["hold_psnr"],
                "ssim": metric["hold_ssim"],
                "lpips": metric["hold_lpips"],
                "primitives": metric["num_gaussians"],
                "old_psnr": old_metric["hold_psnr"],
                "old_ssim": old_metric["hold_ssim"],
                "old_lpips": old_metric["hold_lpips"],
                "old_primitives": old_metric["num_gaussians"],
                "delta_psnr": metric["hold_psnr"] - old_metric["hold_psnr"],
                "delta_ssim": metric["hold_ssim"] - old_metric["hold_ssim"],
                "delta_lpips": metric["hold_lpips"] - old_metric["hold_lpips"],
                "delta_primitives": metric["num_gaussians"]
                - old_metric["num_gaussians"],
                "receipt": str(receipt_path),
            }
        )
        visuals.append(root / f"prism3d_{scene}_turtle/blur-aware/blurred_input_vs_output_top3.png")

    csv_path = SUMMARY_ROOT / "prism3d_dilation2_fix_e8_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    panel_width, panel_height = 760, 570
    header_height, label_height = 106, 58
    canvas = Image.new(
        "RGB",
        (1520, header_height + 4 * (panel_height + label_height)),
        "#101214",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 12),
        "PRISM3D-E | Current dilation-2 blur-aware pipeline | 10K",
        fill="white",
        font=font(23),
    )
    draw.text(
        (20, 48),
        "Each panel: RAW blurred input (left) -> reconstruction (right); hold identity hidden",
        fill="#b8c0c8",
        font=font(18),
    )
    for slot, (row, source) in enumerate(zip(rows, visuals)):
        image = Image.open(source).convert("RGB")
        fitted = ImageOps.contain(
            image,
            (panel_width - 20, panel_height - 10),
            Image.Resampling.LANCZOS,
        )
        column, grid_row = slot % 2, slot // 2
        x = column * panel_width + (panel_width - fitted.width) // 2
        y = header_height + grid_row * (panel_height + label_height)
        canvas.paste(fitted, (x, y))
        label = (
            f"{row['scene']}  {row['psnr']:.2f} dB  SSIM {row['ssim']:.4f}  "
            f"LPIPS {row['lpips']:.4f}  Delta {row['delta_psnr']:+.2f} dB"
        )
        draw.text(
            (column * panel_width + 18, y + panel_height + 8),
            label,
            fill="#efb366",
            font=font(17),
        )
    montage_path = SUMMARY_ROOT / "prism3d_dilation2_fix_e8_raw_to_output_montage.png"
    canvas.save(montage_path)

    averages = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "old_psnr",
            "old_ssim",
            "old_lpips",
            "primitives",
            "old_primitives",
            "delta_psnr",
            "delta_ssim",
            "delta_lpips",
            "delta_primitives",
        )
    }
    (SUMMARY_ROOT / "summary.json").write_text(
        json.dumps({"rows": rows, "averages": averages}, indent=2) + "\n"
    )
    print(json.dumps(averages, sort_keys=True))
    print(csv_path)
    print(montage_path)


if __name__ == "__main__":
    main()
