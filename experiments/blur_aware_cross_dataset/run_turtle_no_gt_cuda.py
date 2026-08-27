#!/usr/bin/env python3
"""Run the released Turtle no-GT inference path on the visible CUDA device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--tile-overlap", type=int, default=128)
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

    def load_model_cuda(path, model):
        if not torch.cuda.is_available():
            raise RuntimeError("Turtle CUDA runner requires a visible CUDA device")
        device = torch.device("cuda:0")
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["params"], strict=True)
        model = model.to(device).eval()
        print(f"> Loaded Turtle on {torch.cuda.get_device_name(0)}", flush=True)
        return model, device

    inference.load_model = load_model_cuda
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
