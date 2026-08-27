#!/usr/bin/env python3
"""Measure how much TUM PSNR is limited by scalar exposure mismatch.

The fitted metrics are diagnostic upper bounds, not admissible benchmark scores:
the affine parameters are fitted directly against the evaluation references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optgs.dataset.colmap.utils import Parser
from optgs.experimental.api import OptGS
from optgs.model.ply_export import load_gaussians_ply

from experiments.blur_aware_cross_dataset.run_cross_dataset import (
    build_views,
    collect_scene,
    read_hold,
    resolve_scene_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tum_fr2_xyz")
    parser.add_argument(
        "--scene-config",
        default=str(Path(__file__).resolve().parent / "scenes.json"),
    )
    parser.add_argument("--ply", required=True)
    parser.add_argument(
        "--checkpoint",
        default=(
            "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
            "checkpoints/epoch_5-step_50000.ckpt"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fit_scalar_affine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = prediction[valid]
    y = target[valid]
    x_centered = x - x.mean()
    gain = (x_centered * (y - y.mean())).sum() / x_centered.square().sum().clamp_min(1e-12)
    gain = gain.clamp(0.25, 4.0)
    bias = (y.mean() - gain * x.mean()).clamp(-0.5, 0.5)
    return gain, bias


def psnr_per_view(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.expand_as(prediction).to(prediction.dtype)
    mse = ((prediction - target).square() * valid).sum(dim=(1, 2, 3))
    mse = mse / valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def ecc_euclidean_upper_bound(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[list[float], list[list[float]], int]:
    scores, transforms, failures = [], [], 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    for pred_view, target_view, mask_view in zip(prediction, target, valid_mask):
        pred_np = pred_view.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        target_np = target_view.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        pred_gray = cv2.cvtColor(pred_np, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(
                target_gray,
                pred_gray,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                inputMask=mask_view[0].detach().cpu().numpy().astype(np.uint8),
            )
        except cv2.error:
            failures += 1
            warp = np.eye(2, 3, dtype=np.float32)
        aligned = cv2.warpAffine(
            pred_np,
            warp,
            (pred_np.shape[1], pred_np.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        support = cv2.warpAffine(
            np.ones(pred_np.shape[:2], dtype=np.uint8),
            warp,
            (pred_np.shape[1], pred_np.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        valid = mask_view[0].detach().cpu().numpy().astype(bool) & support
        mse = float(np.square(aligned[valid] - target_np[valid]).mean())
        scores.append(float(-10.0 * np.log10(max(mse, 1e-12))))
        transforms.append(warp.reshape(-1).astype(float).tolist())
    return scores, transforms, failures


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configs = json.loads(Path(args.scene_config).read_text())
    cfg = configs[args.scene]
    parser = Parser(cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False)
    optimization, evaluation, _, _ = resolve_scene_indices(
        parser, cfg, Path(cfg["data_dir"]), read_hold(Path(cfg["data_dir"]))
    )
    del optimization
    scene = collect_scene(parser, cfg, evaluation)
    views = build_views(scene, evaluation, float(parser.scene_scale * 1.1), device)
    model = OptGS(
        checkpoint=args.checkpoint,
        device=device,
        decoder_backend="fastgs",
        num_refine=1,
    )
    gaussians = load_gaussians_ply(args.ply, max_sh_degree=model.sh_degree).to(device)
    h, w = views.image.shape[-2:]
    output = model.decoder.forward(
        gaussians,
        views.extrinsics,
        views.intrinsics,
        views.near,
        views.far,
        image_shape=(h, w),
    )
    prediction = output.color.clamp(0.0, 1.0).flatten(0, 1)
    target_source = views.raw_image if views.raw_image is not None else views.image
    target = target_source.flatten(0, 1)
    training_target = views.image.flatten(0, 1)
    valid_mask = views.valid_mask.flatten(0, 1).bool()

    base = psnr_per_view(prediction, target, valid_mask)
    render_to_training_target = psnr_per_view(
        prediction, training_target, valid_mask
    )
    training_target_to_raw = psnr_per_view(training_target, target, valid_mask)
    global_valid = valid_mask.expand_as(prediction)
    global_gain, global_bias = fit_scalar_affine(prediction, target, global_valid)
    global_corrected = (prediction * global_gain + global_bias).clamp(0.0, 1.0)
    global_psnr = psnr_per_view(global_corrected, target, valid_mask)

    corrected, gains, biases = [], [], []
    for pred_view, target_view, mask_view in zip(prediction, target, valid_mask):
        expanded = mask_view.expand_as(pred_view)
        gain, bias = fit_scalar_affine(pred_view, target_view, expanded)
        corrected.append((pred_view * gain + bias).clamp(0.0, 1.0))
        gains.append(float(gain))
        biases.append(float(bias))
    per_view_psnr = psnr_per_view(torch.stack(corrected), target, valid_mask)
    ecc_psnr, ecc_transforms, ecc_failures = ecc_euclidean_upper_bound(
        prediction, target, valid_mask
    )

    receipt = {
        "status": "diagnostic_upper_bound_not_benchmark_metric",
        "scene": args.scene,
        "ply": str(Path(args.ply).resolve()),
        "evaluation_views": len(evaluation),
        "base_psnr": float(base.mean()),
        "render_to_training_target_psnr": float(render_to_training_target.mean()),
        "training_target_to_raw_psnr": float(training_target_to_raw.mean()),
        "global_scalar_affine_psnr": float(global_psnr.mean()),
        "global_scalar_affine_delta": float(global_psnr.mean() - base.mean()),
        "per_view_scalar_affine_psnr": float(per_view_psnr.mean()),
        "per_view_scalar_affine_delta": float(per_view_psnr.mean() - base.mean()),
        "ecc_euclidean_psnr": float(sum(ecc_psnr) / len(ecc_psnr)),
        "ecc_euclidean_delta": float(sum(ecc_psnr) / len(ecc_psnr) - base.mean()),
        "ecc_failures": ecc_failures,
        "global_gain": float(global_gain),
        "global_bias": float(global_bias),
        "per_view_gain_min_mean_max": [min(gains), sum(gains) / len(gains), max(gains)],
        "per_view_bias_min_mean_max": [min(biases), sum(biases) / len(biases), max(biases)],
        "base_per_view_psnr": [float(value) for value in base],
        "affine_per_view_psnr": [float(value) for value in per_view_psnr],
        "ecc_per_view_psnr": ecc_psnr,
        "ecc_transforms_2x3": ecc_transforms,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
