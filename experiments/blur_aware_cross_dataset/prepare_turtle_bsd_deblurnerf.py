#!/usr/bin/env python3
"""Generate frame-aligned Turtle-BSD teachers for configured scenes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--turtle-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model-type", choices=("t0", "t1"), default="t0")
    parser.add_argument("--checkpoint-state-key", default="params")
    parser.add_argument("--linear-rgb", action="store_true")
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--scene", action="append")
    parser.add_argument(
        "--scene-prefix",
        action="append",
        default=None,
        help="Accepted scene prefix; defaults to motion_ and defocus_.",
    )
    parser.add_argument(
        "--output-naming",
        choices=("source", "index"),
        default="source",
        help="Preserve source names or emit zero-padded sequence indices.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.name))


def load_scenes(
    paths: list[str],
    requested: list[str] | None,
    prefixes: list[str] | None,
) -> dict[str, dict]:
    scenes: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            scenes.update(json.load(handle))
    accepted = tuple(prefixes or ("motion_", "defocus_"))
    scenes = {k: v for k, v in scenes.items() if k.startswith(accepted)}
    if requested:
        names = set(requested)
        missing = names - scenes.keys()
        if missing:
            raise RuntimeError(f"unknown scenes: {sorted(missing)}")
        scenes = {k: v for k, v in scenes.items() if k in names}
    return scenes


def completed(destination: Path, names: list[str]) -> bool:
    marker = destination / ".complete.json"
    return marker.is_file() and [p.name for p in image_files(destination)] == names


def main() -> None:
    args = parse_args()
    scenes = load_scenes(args.scene_config, args.scene, args.scene_prefix)
    if not scenes:
        raise RuntimeError("no Deblur-NeRF scenes selected")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pending: dict[str, tuple[dict, list[Path]]] = {}
    for scene, cfg in scenes.items():
        files = image_files(Path(cfg["raw_dir"]))
        if not files:
            raise RuntimeError(f"no images in {cfg['raw_dir']}")
        output_names = (
            [p.name for p in files]
            if args.output_naming == "source"
            else [f"{index:06d}.png" for index in range(len(files))]
        )
        if completed(output_root / scene, output_names):
            print(f"[skip complete] {scene}", flush=True)
        else:
            if (output_root / scene).exists():
                raise RuntimeError(f"refusing to overwrite incomplete teacher: {output_root / scene}")
            pending[scene] = (cfg, files)
    if not pending:
        return

    with tempfile.TemporaryDirectory(prefix="turtle_bsd_in_") as input_tmp, tempfile.TemporaryDirectory(
        prefix="turtle_bsd_out_"
    ) as output_tmp:
        input_root = Path(input_tmp)
        for scene, (_, files) in pending.items():
            scene_input = input_root / scene
            scene_input.mkdir()
            for index, source in enumerate(files):
                os.symlink(source.resolve(), scene_input / f"{index:06d}{source.suffix.lower()}")
        runner_command = [
                args.python, args.runner,
                "--turtle-repo", args.turtle_repo,
                "--checkpoint", args.checkpoint,
                "--config", args.config,
                "--data-dir", str(input_root),
                "--output-dir", output_tmp,
                "--model-name", "turtle_bsd_deblurnerf",
                "--model-type", args.model_type,
                "--checkpoint-state-key", args.checkpoint_state_key,
                "--tile", "320",
                "--tile-overlap", "128",
            ]
        if args.linear_rgb:
            runner_command.append("--linear-rgb")
        if args.prediction_only:
            runner_command.append("--prediction-only")
        subprocess.run(
            runner_command,
            check=True,
        )
        generated_root = Path(output_tmp) / "turtle_bsd_deblurnerf"
        for scene, (cfg, files) in pending.items():
            predictions = sorted(
                (generated_root / scene).glob("Frame_*_Pred.png"),
                key=lambda p: int(p.name.split("_")[1]),
            )
            if len(predictions) != len(files):
                raise RuntimeError(f"{scene}: expected {len(files)} predictions, got {len(predictions)}")
            staging = output_root / f".{scene}.partial"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            dimensions = set()
            for index, (prediction, source) in enumerate(zip(predictions, files)):
                output_name = (
                    source.name
                    if args.output_naming == "source"
                    else f"{index:06d}.png"
                )
                shutil.copy2(prediction, staging / output_name)
                from PIL import Image
                with Image.open(prediction) as pred, Image.open(source) as raw:
                    dimensions.add((pred.size, raw.size))
                    if pred.size != raw.size:
                        raise RuntimeError(f"{scene}/{source.name}: {pred.size} != {raw.size}")
            (staging / ".complete.json").write_text(
                json.dumps(
                    {
                        "scene": scene,
                        "frames": len(files),
                        "raw_dir": cfg["raw_dir"],
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "checkpoint_sha256": subprocess.check_output(
                            ["sha256sum", args.checkpoint], text=True
                        ).split()[0],
                        "configuration": str(Path(args.config).resolve()),
                        "model_type": args.model_type,
                        "checkpoint_state_key": args.checkpoint_state_key,
                        "linear_rgb": args.linear_rgb,
                        "prediction_only": args.prediction_only,
                        "output_naming": args.output_naming,
                        "dimensions": [list(map(list, pair)) for pair in sorted(dimensions)],
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            staging.rename(output_root / scene)
            print(f"[complete] {scene}: {len(files)} frames", flush=True)


if __name__ == "__main__":
    main()
