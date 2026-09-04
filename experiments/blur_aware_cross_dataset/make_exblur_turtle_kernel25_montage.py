#!/usr/bin/env python3
"""Combine the eight frozen ExBlur Turtle nominal-k25 visual panels."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/srv2/szha0669/blur_slam_exp/outputs")
PRIMARY = ROOT / "exblur8_turtle_step024000_kernel25_nima06_10k_s1"
RETRY = ROOT / "exblur8_turtle_step024000_kernel25_nima06_10k_retry_gpu23_s2"
OUTPUT = ROOT / "visualizations/exblur8_turtle_step024000_k25d2_nima06_10k_s1"
SCENES = ("bench", "camellia", "dragon", "jars", "jars2", "postbox", "stone_lantern", "sunflowers")


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    panel_width, panel_height = 760, 570
    header_height, label_height = 70, 42
    canvas = Image.new("RGB", (2 * panel_width, header_height + 4 * (panel_height + label_height)), "#101214")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 17), "ExBlur official 8 scenes | Turtle step-24K | nominal k25/dilation2 (effective 49x49) | 10K", fill="white", font=font(25))
    for slot, scene in enumerate(SCENES):
        scene_id = f"exblur_{scene}"
        relative = Path(scene_id) / "blur-aware/blurred_input_vs_output_top3.png"
        source = PRIMARY / relative
        if not source.exists():
            source = RETRY / relative
        image = Image.open(source).convert("RGB")
        fitted = ImageOps.contain(image, (panel_width - 20, panel_height - 12), Image.Resampling.LANCZOS)
        column, row = slot % 2, slot // 2
        x = column * panel_width + (panel_width - fitted.width) // 2
        y = header_height + row * (panel_height + label_height)
        canvas.paste(fitted, (x, y))
        draw.text((column * panel_width + 18, y + panel_height + 6), scene, fill="#efb366", font=font(21))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "exblur8_blurred_input_vs_output_top3_montage.png"
    canvas.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
