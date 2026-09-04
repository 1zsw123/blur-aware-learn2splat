#!/usr/bin/env python3
"""Render frozen Learn2Splat results at blurred input camera poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from optgs.dataset.colmap.utils import Parser
from optgs.experimental.api import OptGS
from optgs.experimental.api.integration.inria_bridge import optgs_gaussians_from_ply

try:
    from experiments.blur_aware_cross_dataset.run_cross_dataset import (
        build_views,
        collect_scene,
    )
except ModuleNotFoundError:
    from run_cross_dataset import build_views, collect_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument(
        "--indices",
        help="optional comma-separated dataset indices; preserves this exact order",
    )
    parser.add_argument(
        "--sharp-reference-dir",
        help=(
            "optional sharp-reference directory used only for post-hoc figure "
            "selection; training and rendering are unchanged"
        ),
    )
    parser.add_argument(
        "--blurred-quantile",
        type=float,
        default=0.5,
        help=(
            "with --sharp-reference-dir, retain this most-blurred fraction of "
            "inputs before ranking outputs by sharp-reference PSNR"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
            "checkpoints/epoch_5-step_50000.ckpt"
        ),
    )
    return parser.parse_args()


def laplacian_variance(images: torch.Tensor) -> torch.Tensor:
    gray = (
        0.2989 * images[:, 0:1]
        + 0.5870 * images[:, 1:2]
        + 0.1140 * images[:, 2:3]
    )
    kernel = images.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel, padding=1)
    return response.flatten(1).var(dim=1, unbiased=False)


def to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(
        np.uint8
    )
    return Image.fromarray(array)


def load_sharp_references(
    directory: Path,
    names: list[str],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    images = []
    for name in names:
        matches = sorted(directory.glob(f"{name}.*"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one sharp reference for {name}, found {matches}"
            )
        image = Image.open(matches[0]).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(images).to(device=device, dtype=dtype)


def psnr_per_view(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    mse = (left - right).square().flatten(1).mean(dim=1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configs = json.loads(Path(args.scene_config).read_text())
    cfg = configs[args.scene]
    run_dir = Path(args.run_root) / args.scene / "blur-aware"
    receipt = json.loads((run_dir / "receipt.json").read_text())

    colmap = Parser(
        cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False
    )
    scene = collect_scene(colmap, cfg, receipt["evaluation_indices"])
    evaluation = set(int(i) for i in receipt["evaluation_indices"])
    if args.indices:
        candidates = [int(value) for value in args.indices.split(",")]
        forbidden = evaluation.intersection(candidates)
        if forbidden:
            raise ValueError(f"selected indices include evaluation views: {forbidden}")
    else:
        candidates = [
            int(i)
            for i in receipt["optimization_indices"]
            if int(i) not in evaluation
        ]
    if not candidates:
        raise RuntimeError("no non-evaluation input views are available")

    views = build_views(
        scene,
        candidates,
        float(colmap.scene_scale * 1.1),
        device,
        image_source="raw",
    )
    model = OptGS(
        checkpoint=args.checkpoint,
        device=device,
        decoder_backend="fastgs",
        num_refine=1,
    )
    gaussians = optgs_gaussians_from_ply(
        run_dir / "point_cloud.ply",
        sh_degree=model.sh_degree,
        device=device,
        dtype=model.dtype,
    )
    model.initialize_from_tensors(gaussians, views)
    height, width = views.image.shape[-2:]
    output = model.decoder.forward(
        gaussians,
        views.extrinsics,
        views.intrinsics,
        views.near,
        views.far,
        image_shape=(height, width),
    ).color[0].clamp(0, 1)
    raw = views.raw_image[0].clamp(0, 1)
    raw_lap = laplacian_variance(raw)
    output_lap = laplacian_variance(output)
    relative_gain = (output_lap - raw_lap) / raw_lap.clamp_min(1e-8)
    raw_reference_psnr = None
    output_reference_psnr = None
    if args.sharp_reference_dir:
        names = [Path(colmap.image_names[index]).stem for index in candidates]
        sharp = load_sharp_references(
            Path(args.sharp_reference_dir),
            names,
            height,
            width,
            device,
            output.dtype,
        )
        raw_reference_psnr = psnr_per_view(raw, sharp)
        output_reference_psnr = psnr_per_view(output, sharp)
    if args.indices:
        order = list(range(len(candidates)))
    elif args.sharp_reference_dir:
        if not 0.0 < args.blurred_quantile <= 1.0:
            raise ValueError("--blurred-quantile must be in (0, 1]")
        keep = max(args.rows, int(np.ceil(len(candidates) * args.blurred_quantile)))
        blurred = torch.argsort(raw_reference_psnr)[:keep].tolist()
        order = sorted(
            blurred,
            key=lambda index: float(output_reference_psnr[index]),
            reverse=True,
        )
    else:
        order = torch.argsort(relative_gain, descending=True).tolist()
    selected = order[: min(args.rows, len(order))]

    header_height = 52
    label_height = 30
    canvas = Image.new(
        "RGB",
        (2 * width, header_height + len(selected) * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = ImageFont.load_default(size=22)
    draw.text((12, 13), "RAW blurred input", fill="black", font=header_font)
    draw.text(
        (width + 12, 13),
        "Ours deblurred output",
        fill="black",
        font=header_font,
    )
    draw.line((0, header_height - 1, 2 * width, header_height - 1), fill="black")
    records = []
    for row, local_index in enumerate(selected):
        y = header_height + row * (height + label_height)
        canvas.paste(to_pil(raw[local_index]), (0, y))
        canvas.paste(to_pil(output[local_index]), (width, y))
        global_index = candidates[local_index]
        name = Path(colmap.image_names[global_index]).stem
        gain = float(relative_gain[local_index])
        draw.text((5, y + height + 7), f"frame {name} | RAW input", fill="black")
        draw.text(
            (width + 5, y + height + 7),
            f"frame {name} | Ours output",
            fill="black",
        )
        records.append(
            {
                "name": name,
                "dataset_index": global_index,
                "raw_laplacian": float(raw_lap[local_index]),
                "output_laplacian": float(output_lap[local_index]),
                "relative_laplacian_gain": gain,
                "raw_reference_psnr": (
                    float(raw_reference_psnr[local_index])
                    if raw_reference_psnr is not None
                    else None
                ),
                "output_reference_psnr": (
                    float(output_reference_psnr[local_index])
                    if output_reference_psnr is not None
                    else None
                ),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    candidates_records = []
    for local_index, global_index in enumerate(candidates):
        candidates_records.append(
            {
                "name": Path(colmap.image_names[global_index]).stem,
                "dataset_index": global_index,
                "raw_laplacian": float(raw_lap[local_index]),
                "output_laplacian": float(output_lap[local_index]),
                "relative_laplacian_gain": float(relative_gain[local_index]),
                "raw_reference_psnr": (
                    float(raw_reference_psnr[local_index])
                    if raw_reference_psnr is not None
                    else None
                ),
                "output_reference_psnr": (
                    float(output_reference_psnr[local_index])
                    if output_reference_psnr is not None
                    else None
                ),
            }
        )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {"scene": args.scene, "rows": records, "candidates": candidates_records},
            indent=2,
        )
        + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
