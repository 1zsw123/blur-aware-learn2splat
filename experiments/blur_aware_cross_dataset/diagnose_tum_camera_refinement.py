#!/usr/bin/env python3
"""Refine TUM train-view cameras against frozen Gaussian geometry.

This is a bounded diagnostic. It never edits the source PLY and reports both
pose displacement and RAW-reference metrics so a large gain cannot be hidden
behind unconstrained camera motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optgs.dataset.colmap.utils import Parser
from optgs.experimental.api import OptGS
from optgs.model.types import Gaussians
from optgs.model.ply_export import load_gaussians_ply
from optgs.scene_trainer.common.gaussians import build_covariance

from experiments.blur_aware_cross_dataset.run_cross_dataset import (
    build_views,
    collect_scene,
    read_hold,
    resolve_scene_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tum_fr2_xyz")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pose-lr", type=float, default=1e-4)
    parser.add_argument("--exposure-lr", type=float, default=1e-3)
    parser.add_argument("--image-warp-lr", type=float, default=0.0)
    parser.add_argument("--pose-prior", type=float, default=1e-4)
    parser.add_argument("--pose-smoothness", type=float, default=1e-3)
    parser.add_argument("--sampling", choices=("random", "cyclic"), default="random")
    parser.add_argument("--retain-best", action="store_true")
    parser.add_argument(
        "--gradient-route",
        choices=("camera", "gaussian_equivalent"),
        default="gaussian_equivalent",
    )
    parser.add_argument(
        "--decoder-backend",
        choices=("checkpoint", "inria", "fastgs"),
        default="fastgs",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--scene-config",
        default=str(Path(__file__).resolve().parent / "scenes.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
            "checkpoints/epoch_5-step_50000.ckpt"
        ),
    )
    return parser.parse_args()


def se3_delta(parameters: torch.Tensor) -> torch.Tensor:
    omega, translation = parameters[:, :3], parameters[:, 3:]
    zero = torch.zeros_like(omega[:, 0])
    skew = torch.stack(
        (
            zero,
            -omega[:, 2],
            omega[:, 1],
            omega[:, 2],
            zero,
            -omega[:, 0],
            -omega[:, 1],
            omega[:, 0],
            zero,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    rotation = torch.matrix_exp(skew)
    upper = torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros((len(parameters), 1, 4), device=parameters.device)
    bottom[:, 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=1)


def psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.expand_as(prediction).to(prediction.dtype)
    mse = ((prediction - target).square() * valid).sum(dim=(1, 2, 3))
    mse = mse / valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def apply_image_plane_correction(images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Apply a differentiable per-view SE(2) residual calibration."""
    angle, shift_x, shift_y = parameters.unbind(dim=-1)
    cosine, sine = angle.cos(), angle.sin()
    theta = torch.stack(
        (cosine, -sine, shift_x, sine, cosine, shift_y), dim=-1
    ).reshape(-1, 2, 3)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def transform_gaussians_for_camera_delta(
    gaussians: Gaussians,
    covariance: torch.Tensor,
    base_extrinsic: torch.Tensor,
    camera_delta: torch.Tensor,
) -> Gaussians:
    """Route a camera-local perturbation through FastGS Gaussian gradients.

    Extrinsics are camera-to-world matrices. Rendering ``C @ D`` is equivalent
    to rendering ``C`` after transforming world geometry by
    ``H = C @ inv(D) @ inv(C)``. FastGS differentiates Gaussian means/covariance
    but not camera matrices, so this preserves the image-space camera effect
    while exposing gradients to the six camera parameters.
    """
    world_transform = base_extrinsic @ torch.linalg.inv(camera_delta) @ torch.linalg.inv(
        base_extrinsic
    )
    rotation = world_transform[:3, :3]
    translation = world_transform[:3, 3]
    means = gaussians.means @ rotation.transpose(0, 1) + translation
    covariances = rotation @ covariance @ rotation.transpose(0, 1)
    return Gaussians(
        means=means,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities,
        scales=gaussians.scales,
        rotations_unnorm=gaussians.rotations_unnorm,
        rotations=gaussians.rotations,
        covariances=covariances,
        stores_activated=gaussians.stores_activated,
        nr_valid=gaussians.nr_valid,
    )


def render_selected(
    model: OptGS,
    gaussians: Gaussians,
    covariance: torch.Tensor,
    base_extrinsics: torch.Tensor,
    views,
    pose: torch.Tensor,
    selected: torch.Tensor,
    image_shape: tuple[int, int],
    gradient_route: str,
) -> torch.Tensor:
    images = []
    deltas = se3_delta(pose[selected])
    for local_index, view_index in enumerate(selected.tolist()):
        if gradient_route == "gaussian_equivalent":
            transformed = transform_gaussians_for_camera_delta(
                gaussians,
                covariance,
                base_extrinsics[0, view_index],
                deltas[local_index],
            )
            extrinsic = base_extrinsics[:, view_index : view_index + 1]
        else:
            transformed = gaussians
            extrinsic = base_extrinsics[:, view_index : view_index + 1] @ deltas[
                local_index
            ].view(1, 1, 4, 4)
        output = model.decoder.forward(
            transformed,
            extrinsic,
            views.intrinsics[:, view_index : view_index + 1],
            views.near[:, view_index : view_index + 1],
            views.far[:, view_index : view_index + 1],
            image_shape=image_shape,
        )
        images.append(output.color.flatten(0, 1))
    return torch.cat(images)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    cfg = json.loads(Path(args.scene_config).read_text())[args.scene]
    parser = Parser(cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False)
    _, evaluation, _, _ = resolve_scene_indices(
        parser, cfg, Path(cfg["data_dir"]), read_hold(Path(cfg["data_dir"]))
    )
    scene = collect_scene(parser, cfg, evaluation)
    views = build_views(scene, evaluation, float(parser.scene_scale * 1.1), device)
    model = OptGS(
        checkpoint=args.checkpoint,
        device=device,
        decoder_backend=(
            None if args.decoder_backend == "checkpoint" else args.decoder_backend
        ),
        num_refine=1,
    )
    gaussians = load_gaussians_ply(args.ply, max_sh_degree=model.sh_degree).to(device)
    rotations = F.normalize(gaussians.rotations_unnorm, dim=-1)
    covariance = build_covariance(gaussians.scales, rotations)
    if args.gradient_route == "gaussian_equivalent":
        if not hasattr(model.decoder.cfg, "use_covariances"):
            raise RuntimeError("Selected decoder cannot consume precomputed covariances")
        model.decoder.cfg.use_covariances = True
    base_extrinsics = views.extrinsics.detach()
    count = base_extrinsics.shape[1]
    pose = torch.nn.Parameter(torch.zeros((count, 6), device=device))
    exposure = torch.nn.Parameter(torch.zeros((count, 2), device=device))
    image_warp = torch.nn.Parameter(torch.zeros((count, 3), device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [pose], "lr": args.pose_lr},
            {"params": [exposure], "lr": args.exposure_lr},
            {"params": [image_warp], "lr": args.image_warp_lr},
        ]
    )
    target = views.image.flatten(0, 1)
    raw = views.raw_image.flatten(0, 1)
    mask = views.valid_mask.flatten(0, 1).bool()
    h, w = target.shape[-2:]
    best_photo = torch.full((count,), float("inf"), device=device)
    best_pose = torch.zeros_like(pose)
    best_exposure = torch.zeros_like(exposure)
    best_image_warp = torch.zeros_like(image_warp)

    for step in range(args.steps):
        if args.sampling == "cyclic":
            selected = (torch.arange(args.batch_size, device=device) + step * args.batch_size) % count
        else:
            selected = torch.randperm(count, device=device)[: args.batch_size]
        prediction = render_selected(
            model,
            gaussians,
            covariance,
            base_extrinsics,
            views,
            pose,
            selected,
            (h, w),
            args.gradient_route,
        )
        prediction = apply_image_plane_correction(prediction, image_warp[selected])
        gain = exposure[selected, 0].exp()[:, None, None, None]
        bias = exposure[selected, 1, None, None, None]
        prediction = (prediction * gain + bias).clamp(0.0, 1.0)
        valid = mask[selected].expand_as(prediction).to(prediction.dtype)
        per_view_photo = (
            F.smooth_l1_loss(prediction, target[selected], reduction="none") * valid
        ).sum(dim=(1, 2, 3))
        per_view_photo = per_view_photo / valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        if args.retain_best:
            improved = per_view_photo.detach() < best_photo[selected]
            improved_indices = selected[improved]
            best_photo[improved_indices] = per_view_photo.detach()[improved]
            best_pose[improved_indices] = pose.detach()[improved_indices]
            best_exposure[improved_indices] = exposure.detach()[improved_indices]
            best_image_warp[improved_indices] = image_warp.detach()[improved_indices]
        photo = per_view_photo.mean()
        regularization = args.pose_prior * pose.square().mean()
        if count > 1:
            regularization = regularization + args.pose_smoothness * (
                pose[1:] - pose[:-1]
            ).square().mean()
        loss = photo + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            print(
                f"pose_grad_abs_max={float(pose.grad.abs().max()):.9g}",
                flush=True,
            )
        optimizer.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"step={step + 1} loss={float(loss):.8f}", flush=True)

    evaluation_pose = best_pose if args.retain_best else pose.detach()
    evaluation_exposure = best_exposure if args.retain_best else exposure.detach()
    evaluation_image_warp = best_image_warp if args.retain_best else image_warp.detach()

    predictions = []
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            selected = torch.arange(start, stop, device=device)
            rendered = render_selected(
                model,
                gaussians,
                covariance,
                base_extrinsics,
                views,
                evaluation_pose,
                selected,
                (h, w),
                args.gradient_route,
            )
            rendered = apply_image_plane_correction(
                rendered, evaluation_image_warp[selected]
            )
            gain = evaluation_exposure[selected, 0].exp()[:, None, None, None]
            bias = evaluation_exposure[selected, 1, None, None, None]
            predictions.append((rendered * gain + bias).clamp(0.0, 1.0))
        prediction = torch.cat(predictions)
        raw_psnr = psnr(prediction, raw, mask)
        target_psnr = psnr(prediction, target, mask)
        rotation_deg = evaluation_pose[:, :3].norm(dim=-1) * (180.0 / torch.pi)
        translation = evaluation_pose[:, 3:].norm(dim=-1)
        image_rotation_deg = evaluation_image_warp[:, 0].abs() * (180.0 / torch.pi)
        image_shift_x_px = evaluation_image_warp[:, 1].abs() * (w / 2.0)
        image_shift_y_px = evaluation_image_warp[:, 2].abs() * (h / 2.0)
        receipt = {
            "status": "train_view_camera_refinement_diagnostic",
            "scene": args.scene,
            "gradient_route": args.gradient_route,
            "steps": args.steps,
            "sampling": args.sampling,
            "retain_best": args.retain_best,
            "raw_psnr": float(raw_psnr.mean()),
            "training_target_psnr": float(target_psnr.mean()),
            "prediction_min_mean_max": [
                float(prediction.min()), float(prediction.mean()), float(prediction.max())
            ],
            "rotation_deg_min_mean_max": [
                float(rotation_deg.min()), float(rotation_deg.mean()), float(rotation_deg.max())
            ],
            "translation_min_mean_max": [
                float(translation.min()), float(translation.mean()), float(translation.max())
            ],
            "image_rotation_deg_abs_min_mean_max": [
                float(image_rotation_deg.min()), float(image_rotation_deg.mean()), float(image_rotation_deg.max())
            ],
            "image_shift_x_px_abs_min_mean_max": [
                float(image_shift_x_px.min()), float(image_shift_x_px.mean()), float(image_shift_x_px.max())
            ],
            "image_shift_y_px_abs_min_mean_max": [
                float(image_shift_y_px.min()), float(image_shift_y_px.mean()), float(image_shift_y_px.max())
            ],
            "exposure_log_gain_min_mean_max": [
                float(evaluation_exposure[:, 0].min()), float(evaluation_exposure[:, 0].mean()), float(evaluation_exposure[:, 0].max())
            ],
            "exposure_bias_min_mean_max": [
                float(evaluation_exposure[:, 1].min()), float(evaluation_exposure[:, 1].mean()), float(evaluation_exposure[:, 1].max())
            ],
        }
        if args.preview:
            preview = torch.cat((raw[0], prediction[0]), dim=2)
            preview = (preview.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
            import cv2

            preview_path = Path(args.preview)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write preview {preview_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.pt")
    torch.save(
        {
            "scene": args.scene,
            "gradient_route": args.gradient_route,
            "pose": evaluation_pose.cpu(),
            "exposure": evaluation_exposure.cpu(),
            "image_warp": evaluation_image_warp.cpu(),
        },
        state_path,
    )
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    receipt["state_path"] = str(state_path)
    receipt["state_sha256"] = state_sha256
    output_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
