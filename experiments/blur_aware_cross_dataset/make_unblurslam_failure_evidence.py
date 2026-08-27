#!/usr/bin/env python3
"""Build a paper-style visual audit of three Unblur-SLAM failure modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


BASE = Path("/srv2/szha0669/blur_slam_exp")
OUT = BASE / "outputs/viz_unblurslam_failure_evidence"


def load(path: Path, size=(648, 484)) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(left - right), axis=2)


def show(ax, image, title, box=None):
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if box is not None:
        x, y, w, h = box
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ef4444", linewidth=2.2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw_55 = load(BASE / "data/scannet/scene0002_01/color/000055.jpg")
    restored_55 = load(BASE / "repos/EVSSM/results_final_2/scene0002_01/GoPro/000055.png")
    diff_55 = difference(raw_55, restored_55)

    raw_158 = load(BASE / "data/scannet_halfres_preprocess/scene0031_00_full/raw/000158.png")
    restored_158 = load(
        BASE / "data/evssm_scannet_halfres/scene0031_00_full_unblurslam_corrected/images/000158.png"
    )
    render_158 = load(
        BASE
        / "outputs/render/scannet_scene0031_00_full_rgbdpose_poseqc2614_unblurslam_ar20_earlyrel_equiv10k_gpu2_ramcache_resume10k/scene0031_00/000158.png"
    )

    fig = plt.figure(figsize=(15.5, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.9))

    box_55 = (55, 345, 300, 125)
    show(fig.add_subplot(grid[0, 0]), raw_55, "(a) RAW input", box_55)
    show(fig.add_subplot(grid[0, 1]), restored_55, "Fixed restored target", box_55)
    ax = fig.add_subplot(grid[0, 2])
    heat = ax.imshow(diff_55, cmap="magma", vmin=0, vmax=np.quantile(diff_55, 0.995))
    ax.add_patch(Rectangle(box_55[:2], box_55[2], box_55[3], fill=False, edgecolor="#22c55e", linewidth=2.2))
    ax.set_title("Restoration residual |teacher - RAW|", fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02)

    show(fig.add_subplot(grid[1, 0]), raw_158, "(b) RAW input")
    show(fig.add_subplot(grid[1, 1]), restored_158, "Restored supervision")
    show(fig.add_subplot(grid[1, 2]), render_158, "Unblur-SLAM scene render", (555, 30, 90, 180))

    scene_labels = ["scene0009", "scene0024", "scene0031", "scene0200"]
    teacher_gain = np.array([0.0097158, 0.0228677, 0.0357738, 0.0391169])
    scene_gain = np.array([0.0279813, -0.0129975, -0.0356035, 0.0299659])
    x = np.arange(len(scene_labels))
    width = 0.34
    ax = fig.add_subplot(grid[2, :2])
    ax.bar(x - width / 2, teacher_gain, width, label="Restored - RAW NIMA", color="#0ea5e9")
    ax.bar(x + width / 2, scene_gain, width, label="Scene render - restored NIMA", color="#ef4444")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, scene_labels)
    ax.set_ylabel("NIMA difference")
    ax.set_title("(c) Restoration gain does not determine scene capacity or final quality", fontweight="bold")
    ax.legend(frameon=False, ncols=2, loc="lower left")
    ax.grid(axis="y", alpha=0.2)

    ax = fig.add_subplot(grid[2, 2])
    ax.axis("off")
    schedule = (
        "Unblur-SLAM fixed capacity schedule\n\n"
        "densify_from_iter   = 500\n"
        "densify_until_iter  = 15,000\n"
        "densification_interval = 100\n"
        "gradient threshold  = 0.0002\n\n"
        "Same schedule for every frame/scene.\n"
        "No EVSSM reliability, blur kernel,\n"
        "mask, or sharpness-surplus signal\n"
        "enters the densification gate."
    )
    ax.text(
        0.04,
        0.96,
        schedule,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )

    fig.suptitle(
        "Observed Unblur-SLAM Failure Evidence",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "(a) Direct observation: restoration introduces localized chromatic holes.  "
        "(b) Direct observation: the reconstructed scene contains boundary holes/distortion.  "
        "(c) Code-level fact: capacity scheduling is fixed and independent of blur evidence; bars show scene-dependent outcomes.",
        ha="center",
        fontsize=10.5,
    )

    png = OUT / "unblurslam_three_failure_modes.png"
    pdf = OUT / "unblurslam_three_failure_modes.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
