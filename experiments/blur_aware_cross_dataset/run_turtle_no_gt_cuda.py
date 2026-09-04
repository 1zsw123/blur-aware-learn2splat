#!/usr/bin/env python3
"""Run the released Turtle no-GT inference path on the visible CUDA device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turtle-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-type", choices=("t0", "t1"), required=True)
    parser.add_argument(
        "--checkpoint-state-key",
        default="params",
        help="State-dict key in the checkpoint (released models use params).",
    )
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument(
        "--linear-rgb",
        action="store_true",
        help="Match the staged training contract: sRGB input -> linear model -> sRGB output.",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Save only restored PNGs; avoid matplotlib figures and input copies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.turtle_repo).resolve()
    sys.path.insert(0, str(repo / "basicsr"))
    sys.path.insert(0, str(repo))
    from basicsr import inference_no_ground_truth as inference

    original_savefig = inference.plt.savefig

    def savefig_and_close(*save_args, **save_kwargs):
        try:
            return original_savefig(*save_args, **save_kwargs)
        finally:
            inference.plt.close()

    inference.plt.savefig = savefig_and_close

    if args.linear_rgb:
        original_getitem = inference.VideoLoader.__getitem__
        original_patched = inference.run_inference_patched

        def srgb_to_linear(value):
            return torch.where(
                value <= 0.04045,
                value / 12.92,
                ((value + 0.055) / 1.055).pow(2.4),
            )

        def linear_to_srgb(value):
            return torch.where(
                value <= 0.0031308,
                value * 12.92,
                1.055 * value.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
            ).clamp(0.0, 1.0)

        def getitem_linear(loader, index):
            target, image = original_getitem(loader, index)
            return target, srgb_to_linear(image)

        def run_patched_linear(*run_args, **run_kwargs):
            restored, k_cache, v_cache = original_patched(*run_args, **run_kwargs)
            return linear_to_srgb(restored), k_cache, v_cache

        inference.VideoLoader.__getitem__ = getitem_linear
        inference.run_inference_patched = run_patched_linear

    def load_model_cuda(path, model):
        if not torch.cuda.is_available():
            raise RuntimeError("Turtle CUDA runner requires a visible CUDA device")
        device = torch.device("cuda:0")
        payload = torch.load(path, map_location="cpu")
        if args.checkpoint_state_key not in payload:
            raise KeyError(
                f"checkpoint has no {args.checkpoint_state_key!r}; "
                f"available keys: {sorted(payload)}"
            )
        model.load_state_dict(payload[args.checkpoint_state_key], strict=True)
        model = model.to(device).eval()
        print(f"> Loaded Turtle on {torch.cuda.get_device_name(0)}", flush=True)
        return model, device

    inference.load_model = load_model_cuda

    if args.prediction_only:
        def run_prediction_only(
            video_name, test_loader, model, device, model_name, save_img,
            do_patched, image_out_path, tile, tile_overlap, model_type,
        ):
            previous_frame = None
            k_cache = v_cache = None
            destination = Path(image_out_path) / model_name / video_name
            destination.mkdir(parents=True, exist_ok=True)
            for index in range(len(test_loader.dataset)):
                current_frame = test_loader.dataset[index][1]
                if previous_frame is None:
                    previous_frame = current_frame
                _, height, width = current_frame.shape
                restored, k_cache, v_cache = inference.run_inference_patched(
                    previous_frame.unsqueeze(0),
                    current_frame.unsqueeze(0),
                    model,
                    device,
                    tile=tile,
                    tile_overlap=tile_overlap,
                    prev_patch_dict_k=k_cache,
                    prev_patch_dict_v=v_cache,
                    model_type=model_type,
                )
                restored = restored.squeeze(0)[:, :height, :width]
                array = (
                    restored.permute(1, 2, 0).detach().cpu().numpy() * 255.0
                ).round().clip(0, 255).astype(np.uint8)
                cv2.imwrite(
                    str(destination / f"Frame_{index + 1}_Pred.png"),
                    cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
                )
                previous_frame = current_frame
            return None, None

        inference.run_inference = run_prediction_only

    inference.main(
        model_path=args.checkpoint,
        model_name=args.model_name,
        data_dir=args.data_dir,
        config_file=args.config,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
        save_image=True,
        model_type=args.model_type,
        do_pacthes=True,
        image_out_path=args.output_dir,
    )


if __name__ == "__main__":
    main()
