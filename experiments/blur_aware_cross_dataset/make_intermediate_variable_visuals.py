#!/usr/bin/env python3
"""Visualize restoration inputs, latent Gaussians, and capacity dynamics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-ours-figure", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--evssm-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=30_000)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def image_for_stem(directory: Path, stem: str) -> Image.Image:
    matches = sorted(
        path
        for path in directory.glob(f"{stem}.*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one image for {directory}/{stem}, found {matches}")
    return Image.open(matches[0]).convert("RGB")


def extract_ours_rows(
    figure_path: Path,
    row_count: int,
    image_size: tuple[int, int],
) -> list[Image.Image]:
    figure = Image.open(figure_path).convert("RGB")
    width, height = image_size
    header_height = 52
    label_height = 30
    expected = (2 * width, header_height + row_count * (height + label_height))
    if figure.size != expected:
        raise RuntimeError(f"unexpected source figure size {figure.size}, expected {expected}")
    return [
        figure.crop(
            (
                width,
                header_height + row * (height + label_height),
                2 * width,
                header_height + row * (height + label_height) + height,
            )
        )
        for row in range(row_count)
    ]


def make_restoration_panel(
    *,
    names: list[str],
    raw_dir: Path,
    middle_dir: Path,
    middle_title: str,
    ours_rows: list[Image.Image],
    output: Path,
) -> None:
    raw = [image_for_stem(raw_dir, name) for name in names]
    middle = [image_for_stem(middle_dir, name) for name in names]
    width, height = raw[0].size
    for images in (raw, middle, ours_rows):
        if any(image.size != (width, height) for image in images):
            raise RuntimeError("panel images do not share one resolution")
    header_height, label_height = 54, 30
    canvas = Image.new(
        "RGB",
        (3 * width, header_height + len(names) * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    titles = ("RAW blurred input", middle_title, "Ours deblurred output")
    for column, title in enumerate(titles):
        draw.text((column * width + 12, 13), title, fill="black", font=font(21))
    draw.line((0, header_height - 1, canvas.width, header_height - 1), fill="black")
    for row, name in enumerate(names):
        y = header_height + row * (height + label_height)
        for column, image in enumerate((raw[row], middle[row], ours_rows[row])):
            canvas.paste(image, (column * width, y))
            draw.text(
                (column * width + 5, y + height + 7),
                f"frame {name} | {titles[column]}",
                fill="black",
                font=font(12),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def visualize_latent_scene(ply_path: Path, output: Path, max_points: int) -> dict:
    vertex = PlyData.read(ply_path)["vertex"].data
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    dc = np.column_stack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]))
    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    xyz, rgb, opacity = xyz[finite], rgb[finite], opacity[finite]
    threshold = np.quantile(opacity, 0.35)
    keep = np.flatnonzero(opacity >= threshold)
    rng = np.random.default_rng(20260904)
    if len(keep) > max_points:
        probability = opacity[keep] / opacity[keep].sum()
        keep = rng.choice(keep, size=max_points, replace=False, p=probability)
    xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]

    center = np.median(xyz, axis=0)
    centered = xyz - center
    covariance = np.cov(centered, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    aligned = centered @ vectors[:, ::-1]
    limits = np.quantile(np.abs(aligned), 0.985, axis=0)
    within = (np.abs(aligned) <= limits).all(axis=1)
    aligned, rgb, opacity = aligned[within], rgb[within], opacity[within]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    pairs = ((0, 1), (0, 2), (1, 2))
    labels = (("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3"))
    order = np.argsort(aligned[:, 2])
    for axis, pair, axis_labels in zip(axes, pairs, labels):
        axis.scatter(
            aligned[order, pair[0]],
            aligned[order, pair[1]],
            c=rgb[order],
            s=0.45 + 1.8 * opacity[order],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Latent 3D Gaussian scene | {len(vertex):,} primitives | colored Gaussian-center projections",
        fontsize=16,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "total_primitives": int(len(vertex)),
        "finite_primitives": int(finite.sum()),
        "displayed_primitives": int(len(aligned)),
        "opacity_selection_quantile": 0.35,
        "coordinate_alignment": "PCA",
        "visualization": "Gaussian-center projections, not a camera render",
    }


def visualize_capacity(receipt_path: Path, output: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    events = sorted(receipt["capacity_events"], key=lambda event: int(event["step"]))
    steps = np.asarray([int(event["step"]) for event in events])
    counts = np.asarray([int(event["num_gaussians"]) for event in events])
    cloned = np.asarray([int(event["cloned"]) for event in events])
    split = np.asarray([int(event["split"]) for event in events])
    pruned = np.asarray([int(event["pruned"]) for event in events])
    quality = np.asarray([float(event["blur_quality_reward"]) for event in events])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, facecolor="white")
    axes[0].plot(steps, counts, color="#167d9a", linewidth=2.3)
    axes[0].fill_between(steps, counts, color="#8ed2df", alpha=0.35)
    axes[0].set_ylabel("Gaussian primitives")
    axes[0].set_title(f"Gaussian capacity control | final recorded count: {counts[-1]:,}")
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, cloned, label="clone", color="#2a9d8f", linewidth=1.4)
    axes[1].plot(steps, split, label="split", color="#e9a23b", linewidth=1.4)
    axes[1].plot(steps, pruned, label="prune", color="#d14b4b", linewidth=1.4)
    axes[1].set_ylabel("Primitives / action")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(alpha=0.2)

    axes[2].axhline(0.0, color="#444444", linewidth=0.8)
    axes[2].plot(steps, quality, color="#7253a3", linewidth=1.4)
    axes[2].fill_between(steps, 0.0, quality, where=quality >= 0, color="#58a65c", alpha=0.3)
    axes[2].fill_between(steps, 0.0, quality, where=quality < 0, color="#d14b4b", alpha=0.3)
    axes[2].set_ylabel("Blur-quality reward")
    axes[2].set_xlabel("Training step")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "events": len(events),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first_recorded_primitives": int(counts[0]),
        "last_recorded_primitives": int(counts[-1]),
        "max_recorded_primitives": int(counts.max()),
        "total_cloned": int(cloned.sum()),
        "total_split": int(split.sum()),
        "total_pruned": int(pruned.sum()),
    }


def main() -> None:
    args = parse_args()
    configs = json.loads(args.scene_config.read_text())
    cfg = configs[args.scene]
    selection = json.loads(args.selection_json.read_text())
    names = [str(row["name"]) for row in selection["rows"]]
    raw_dir = Path(cfg["raw_dir"])
    teacher_dir = Path(cfg["evssm_dir"])
    first_raw = image_for_stem(raw_dir, names[0])
    ours_rows = extract_ours_rows(args.raw_ours_figure, len(names), first_raw.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual_path = args.output_dir / "raw_turtle_teacher_ours.png"
    make_restoration_panel(
        names=names,
        raw_dir=raw_dir,
        middle_dir=teacher_dir,
        middle_title="Turtle step-24K teacher (actual)",
        ours_rows=ours_rows,
        output=actual_path,
    )
    evssm_path = None
    if args.evssm_dir is not None:
        evssm_path = args.output_dir / "raw_evssm_baseline_ours.png"
        make_restoration_panel(
            names=names,
            raw_dir=raw_dir,
            middle_dir=args.evssm_dir,
            middle_title="EVSSM baseline (not used by this run)",
            ours_rows=ours_rows,
            output=evssm_path,
        )

    latent_path = args.output_dir / "latent_3d_gaussian_scene.png"
    capacity_path = args.output_dir / "gaussian_capacity_dynamics.png"
    latent = visualize_latent_scene(args.run_dir / "point_cloud.ply", latent_path, args.max_points)
    capacity = visualize_capacity(args.run_dir / "receipt.json", capacity_path)
    metadata = {
        "scene": args.scene,
        "frames": names,
        "raw_dir": str(raw_dir),
        "actual_teacher": "Turtle stage-1 step-024000",
        "actual_teacher_dir": str(teacher_dir),
        "evssm_role": "comparison baseline only; not an intermediate of this accepted run",
        "evssm_dir": str(args.evssm_dir) if args.evssm_dir else None,
        "actual_restoration_panel": str(actual_path),
        "evssm_comparison_panel": str(evssm_path) if evssm_path else None,
        "latent_scene": latent,
        "capacity": capacity,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
