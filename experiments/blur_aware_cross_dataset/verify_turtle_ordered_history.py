#!/usr/bin/env python3
"""Fail-closed ordered-history verification for a Turtle training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_frames(directory: Path, count: int, crop: int) -> list[torch.Tensor]:
    paths = sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} frames in {directory}, got {len(paths)}")
    frames = []
    for path in paths:
        srgb = pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0
        h, w = srgb.shape[-2:]
        top, left = (h - crop) // 2, (w - crop) // 2
        if top < 0 or left < 0:
            raise RuntimeError(f"{path}: image is smaller than {crop}x{crop}")
        # The training contract applies sRGB -> linear before model and loss.
        linear = torch.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055).pow(2.4),
        )
        frames.append(linear[:, top : top + crop, left : left + crop])
    return frames


def run_sequence(model, frames, *, reset: bool) -> tuple[list[torch.Tensor], list[dict]]:
    outputs, cache_rows = [], []
    k_cache = v_cache = None
    previous = None
    with torch.no_grad():
        for index, current in enumerate(frames):
            previous = current if previous is None else previous
            pair = torch.stack((previous, current), dim=0).unsqueeze(0).cuda()
            output, k_new, v_new = model(pair, None if reset else k_cache, None if reset else v_cache)
            if len(k_new) != len(v_new) or not k_new:
                raise RuntimeError("Turtle returned an invalid K/V cache")
            cache_rows.append(
                {
                    "frame": index,
                    "levels": len(k_new),
                    "k_shapes": [list(value.shape) if value is not None else None for value in k_new],
                    "v_shapes": [list(value.shape) if value is not None else None for value in v_new],
                    "finite": all(
                        value is None or bool(torch.isfinite(value).all())
                        for value in [*k_new, *v_new]
                    ),
                }
            )
            if not cache_rows[-1]["finite"]:
                raise RuntimeError(f"non-finite cache at frame {index}")
            outputs.append(output.detach().cpu())
            if not reset:
                k_cache, v_cache = k_new, v_new
            previous = current
    return outputs, cache_rows


def mean_abs(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return float(torch.stack([(a - b).abs().mean() for a, b in zip(left, right)]).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turtle-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--crop", type=int, default=192)
    parser.add_argument("--visual-dir", type=Path)
    args = parser.parse_args()

    sys.path[:0] = [str(args.turtle_repo / "basicsr"), str(args.turtle_repo)]
    from basicsr import inference_no_ground_truth as inference
    from basicsr.utils.options import parse

    payload = torch.load(args.checkpoint, map_location="cpu")
    if payload.get("format") != "unblur_slam.turtle_unblur_stable_checkpoint.v3":
        raise RuntimeError(f"unexpected checkpoint format: {payload.get('format')!r}")
    model = inference.create_video_model(parse(str(args.config), is_train=True), "t1")
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model = model.cuda().eval()

    frames = load_frames(args.frames, args.count, args.crop)
    ordered, ordered_cache = run_sequence(model, frames, reset=False)
    reset, reset_cache = run_sequence(model, frames, reset=True)
    reverse, reverse_cache = run_sequence(model, list(reversed(frames)), reset=False)
    reverse_aligned = list(reversed(reverse))
    if any(tuple(output.shape) != (1, 3, args.crop, args.crop) for output in ordered):
        raise RuntimeError("unexpected Turtle output shape")
    ordered_reset_delta = mean_abs(ordered[1:], reset[1:])
    ordered_reverse_delta = mean_abs(ordered, reverse_aligned)
    if ordered_reset_delta <= 1e-7 or ordered_reverse_delta <= 1e-7:
        raise RuntimeError("history/order behavior is numerically indistinguishable")

    def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value <= 0.0031308,
            value * 12.92,
            1.055 * value.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
        ).clamp(0.0, 1.0)

    def laplacian_variance(value: torch.Tensor) -> float:
        gray = (value * torch.tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        response = F.conv2d(gray[None, None], kernel[None, None], padding=1)
        return float(response.var())

    frame_quality = []
    if args.visual_dir is not None:
        args.visual_dir.mkdir(parents=True, exist_ok=True)
    for index, (source_linear, output_linear) in enumerate(zip(frames, ordered)):
        source = linear_to_srgb(source_linear)
        output = linear_to_srgb(output_linear[0])
        mse = float((source - output).square().mean())
        source_lap, output_lap = laplacian_variance(source), laplacian_variance(output)
        frame_quality.append(
            {
                "frame": index,
                "output_vs_input_psnr": -10.0 * math.log10(max(mse, 1e-12)),
                "input_laplacian": source_lap,
                "output_laplacian": output_lap,
                "relative_laplacian_change": (output_lap - source_lap) / max(source_lap, 1e-12),
            }
        )
        if args.visual_dir is not None:
            to_pil_image(source).save(args.visual_dir / f"frame_{index:02d}_input.png")
            to_pil_image(output).save(args.visual_dir / f"frame_{index:02d}_output.png")

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "format": payload["format"],
        "stage": payload["stage"],
        "step": payload["step"],
        "state_key": "model",
        "strict_load": True,
        "model_type": "t1",
        "ordered_frames": args.count,
        "input_color_space": "linear_rgb_from_srgb",
        "output_shapes": [list(value.shape) for value in ordered],
        "ordered_vs_reset_mean_abs": ordered_reset_delta,
        "ordered_vs_reverse_mean_abs": ordered_reverse_delta,
        "frame_quality": frame_quality,
        "ordered_cache": ordered_cache,
        "reset_cache": reset_cache,
        "reverse_cache": reverse_cache,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
