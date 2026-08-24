#!/usr/bin/env python3
"""Compute LPIPS from a frozen final PLY without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from optgs.dataset.colmap.utils import Parser
from optgs.experimental.api import OptGS
from optgs.experimental.api.integration.inria_bridge import optgs_gaussians_from_ply

try:
    from experiments.blur_aware_cross_dataset.run_cross_dataset import (
        build_views,
        collect_scene,
        read_hold,
        render_metrics,
        resolve_scene_indices,
    )
except ModuleNotFoundError:
    from run_cross_dataset import (
        build_views,
        collect_scene,
        read_hold,
        render_metrics,
        resolve_scene_indices,
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--scene-config", default=str(here / "scenes.json"))
    parser.add_argument(
        "--checkpoint",
        default=(
            "/srv2/szha0669/blur_slam_exp/checkpoints/learn2splat/dense/"
            "checkpoints/epoch_5-step_50000.ckpt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-batch-size", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LPIPS postprocess requires CUDA rendering")

    configs = json.loads(Path(args.scene_config).read_text())
    cfg = configs[args.scene]
    source_dir = Path(args.run_root) / args.scene / "blur-aware"
    ply_path = source_dir / "point_cloud.ply"
    receipt_path = source_dir / "receipt.json"
    output_path = source_dir / "lpips_postprocess.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not ply_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing frozen PLY/receipt in {source_dir}")

    receipt = json.loads(receipt_path.read_text())
    parser = Parser(
        cfg["data_dir"], factor=int(cfg["factor"]), normalize=False, verbose=False
    )
    hold = read_hold(Path(cfg["data_dir"]))
    _, evaluation_indices, evaluation_source, _ = resolve_scene_indices(
        parser, cfg, Path(cfg["data_dir"]), hold
    )
    scene = collect_scene(parser, cfg, evaluation_indices)
    scene_scale = float(parser.scene_scale * 1.1)
    test_views = build_views(scene, evaluation_indices, scene_scale, device)

    optgs = OptGS(
        checkpoint=args.checkpoint,
        device=device,
        decoder_backend="fastgs",
        num_refine=1,
    )
    gaussians = optgs_gaussians_from_ply(
        ply_path,
        sh_degree=optgs.sh_degree,
        device=device,
        dtype=optgs.dtype,
    )
    optgs.initialize_from_tensors(gaussians, test_views)

    import lpips

    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    metrics = render_metrics(
        optgs,
        gaussians,
        test_views,
        lpips_model=lpips_model,
        metric_batch_size=args.metric_batch_size,
    )
    frozen_final = receipt["metrics"][-1]
    psnr_delta = float(metrics["psnr"] - frozen_final["hold_psnr"])
    ssim_delta = float(metrics["ssim"] - frozen_final["hold_ssim"])
    if abs(psnr_delta) > 0.05 or abs(ssim_delta) > 5e-4:
        raise RuntimeError(
            "frozen PLY rerender does not reproduce final metrics: "
            f"PSNR delta={psnr_delta:.6f}, SSIM delta={ssim_delta:.6f}"
        )

    payload = {
        "scene": args.scene,
        "evaluation_source": evaluation_source,
        "evaluation_count": len(evaluation_indices),
        "lpips_network": "alexnet-v0.1",
        "normalize": True,
        "hold_lpips": metrics["lpips"],
        "hold_per_view_lpips": metrics["per_view_lpips"],
        "rerender_psnr": metrics["psnr"],
        "rerender_ssim": metrics["ssim"],
        "frozen_psnr_delta": psnr_delta,
        "frozen_ssim_delta": ssim_delta,
        "source_ply": str(ply_path),
        "source_ply_sha256": sha256(ply_path),
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": sha256(receipt_path),
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
