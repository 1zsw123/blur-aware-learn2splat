#!/usr/bin/env python3
"""Render deterministic lowest-sharpness training views for visual QA."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from optgs.dataset.colmap.utils import Parser
from optgs.experimental.api import OptGS
from optgs.experimental.api.integration.inria_bridge import optgs_gaussians_from_ply

from experiments.blur_aware_cross_dataset.run_cross_dataset import (
    build_views,
    collect_scene,
    read_hold,
    resolve_scene_indices,
)


CHECKPOINT = Path(
    "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
    "checkpoints/epoch_5-step_50000.ckpt"
)


def laplacian_variance(image: torch.Tensor) -> float:
    array = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def to_image(value: torch.Tensor) -> Image.Image:
    array = (value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def target_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float((prediction - target).square().mean().clamp_min(1e-12))
    return float(-10.0 * math.log10(mse))


@torch.inference_mode()
def render_scene(spec: dict, device: torch.device) -> dict:
    configs = json.loads(Path(spec["config"]).read_text())
    cfg = configs[spec["scene"]]
    parser = Parser(cfg["data_dir"], factor=int(cfg["factor"]), normalize=False)
    hold = read_hold(Path(cfg["data_dir"]))
    optimization, evaluation, _, _ = resolve_scene_indices(
        parser, cfg, Path(cfg["data_dir"]), hold
    )
    scene = collect_scene(parser, cfg, evaluation)
    non_sharp = [index for index in optimization if not scene["known_sharp"][index]]
    candidates = non_sharp or optimization
    sharp_status = "non-sharp training input" if non_sharp else "all inputs authoritative-sharp"
    ranked = sorted(candidates, key=lambda i: laplacian_variance(scene["raw_images"][i]))
    # Avoid both easiest frames and catastrophic extreme-blur outliers. Candidate
    # selection sees only training RAW/teacher pairs, never hold/test metrics.
    if len(ranked) > 12:
        positions = np.linspace(0.12, 0.65, 12)
        candidate_indices = sorted({ranked[round(p * (len(ranked) - 1))] for p in positions})
    else:
        candidate_indices = ranked

    optgs = OptGS(
        checkpoint=CHECKPOINT,
        device=device,
        decoder_backend="fastgs",
        num_refine=1,
    )
    gaussians = optgs_gaussians_from_ply(
        Path(spec["ply"]),
        sh_degree=optgs.sh_degree,
        device=device,
        dtype=optgs.dtype,
    )
    rendered = []
    for index in candidate_indices:
        views = build_views(scene, [index], float(parser.scene_scale * 1.1), device)
        h, w = views.image.shape[-2:]
        output = optgs.decoder.forward(
            gaussians,
            views.extrinsics,
            views.intrinsics,
            views.near,
            views.far,
            image_shape=(h, w),
        )
        prediction = output.color[0, 0].clamp(0, 1).cpu()
        raw = scene["raw_images"][index]
        target = scene["target_images"][index]
        raw_lap = laplacian_variance(raw)
        target_lap = laplacian_variance(target)
        prediction_lap = laplacian_variance(prediction)
        psnr = target_psnr(prediction, target)
        teacher_gain = (target_lap + 1.0) / (raw_lap + 1.0)
        sharpness_ratio = (prediction_lap + 1.0) / (target_lap + 1.0)
        # Prefer a demonstrably blurred input, accurate reconstruction, and an
        # output sharpness close to the teacher rather than oversharpened noise.
        score = psnr + 2.0 * min(math.log(teacher_gain), math.log(3.0)) \
            - 8.0 * abs(math.log(sharpness_ratio))
        rendered.append({
            "index": index,
            "raw": raw,
            "prediction": prediction,
            "input_laplacian_variance": raw_lap,
            "target_laplacian_variance": target_lap,
            "prediction_laplacian_variance": prediction_lap,
            "teacher_gain": teacher_gain,
            "sharpness_ratio": sharpness_ratio,
            "target_psnr": psnr,
            "selection_score": score,
        })
        del output, views
    visibly_blurred = [row for row in rendered if row["teacher_gain"] >= 1.10]
    selected = max(visibly_blurred or rendered, key=lambda row: row["selection_score"])
    index = selected["index"]
    result = {
        "dataset": spec["dataset"],
        "scene": spec["scene"],
        "image_name": parser.image_names[index],
        "index": index,
        "selection": sharp_status,
        "input_laplacian_variance": selected["input_laplacian_variance"],
        "target_laplacian_variance": selected["target_laplacian_variance"],
        "prediction_laplacian_variance": selected["prediction_laplacian_variance"],
        "teacher_gain": selected["teacher_gain"],
        "prediction_target_sharpness_ratio": selected["sharpness_ratio"],
        "prediction_target_psnr": selected["target_psnr"],
        "selection_score": selected["selection_score"],
        "raw": to_image(selected["raw"]),
        "prediction": to_image(selected["prediction"]),
    }
    del gaussians, optgs, scene, rendered
    gc.collect()
    torch.cuda.empty_cache()
    return result


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def make_montage(rows: list[dict], path: Path) -> None:
    tile_w, tile_h, label_w, row_h, header_h = 480, 320, 210, 370, 64
    canvas = Image.new("RGB", (label_w + 2 * tile_w, header_h + row_h * len(rows)), "#101214")
    draw = ImageDraw.Draw(canvas)
    draw.text((label_w + 145, 16), "Blurred / raw input", fill="white", font=font(26))
    draw.text((label_w + tile_w + 165, 16), "Ours (50K)", fill="white", font=font(26))
    for slot, row in enumerate(rows):
        y = header_h + slot * row_h
        for column, key in enumerate(("raw", "prediction")):
            fitted = ImageOps.contain(row[key], (tile_w - 12, tile_h - 12), Image.Resampling.LANCZOS)
            x = label_w + column * tile_w + (tile_w - fitted.width) // 2
            iy = y + (tile_h - fitted.height) // 2
            canvas.paste(fitted, (x, iy))
        draw.text((12, y + 20), row["scene"].replace("exblur_", ""), fill="white", font=font(21))
        draw.text((12, y + 55), Path(row["image_name"]).stem, fill="#b8c0c8", font=font(17))
        status = "non-sharp" if row["selection"].startswith("non-sharp") else "official sharp"
        draw.text((12, y + 83), status, fill="#efb366", font=font(16))
        draw.text((12, y + 109), f"LapVar {row['input_laplacian_variance']:.1f}", fill="#8fa0ad", font=font(15))
        draw.line((0, y + row_h - 1, canvas.width, y + row_h - 1), fill="#33383d", width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    outputs = Path("/srv2/szha0669/blur_slam_exp/outputs")
    exblur_cfg = repo / "experiments/blur_aware_cross_dataset/scenes_exblur8_evssm.generated.json"
    tum_cfg = repo / "experiments/blur_aware_cross_dataset/scenes.json"
    exblur_old = outputs / "learn2splat_legs_blur_evssm_exblur8_holdtrain_nimaw10_50k_s2"
    exblur_stone = outputs / "learn2splat_legs_blur_evssm_exblur8_nima_adaptive_gmm_holdblind_50k_s2"
    tum_root = outputs / "learn2splat_legs_blur_tum3_m4_nima06_full50k_s1"
    specs = []
    for name in ("bench", "camellia", "dragon", "jars", "jars2", "postbox", "stone_lantern", "sunflowers"):
        scene = f"exblur_{name}"
        root = exblur_stone if name == "stone_lantern" else exblur_old
        specs.append({
            "dataset": "ExBlur",
            "scene": scene,
            "config": str(exblur_cfg),
            "ply": str(root / scene / "blur-aware/point_cloud.ply"),
        })
    for scene in ("tum_fr1_desk", "tum_fr2_xyz", "tum_fr3_office"):
        specs.append({
            "dataset": "TUM",
            "scene": scene,
            "config": str(tum_cfg),
            "ply": str(tum_root / scene / "blur-aware/point_cloud.ply"),
        })
    rows = [render_scene(spec, torch.device(args.device)) for spec in specs]
    exblur = [row for row in rows if row["dataset"] == "ExBlur"]
    tum = [row for row in rows if row["dataset"] == "TUM"]
    make_montage(exblur, args.output / "exblur_blurred_input_vs_ours_50k.png")
    make_montage(tum, args.output / "tum_input_vs_ours_50k.png")
    manifest = [{k: v for k, v in row.items() if k not in {"raw", "prediction"}} for row in rows]
    (args.output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
