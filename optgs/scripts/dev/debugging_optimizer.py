import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from optgs.misc.io import CustomPath
from optgs.model.types import Gaussians

plt.rcParams.update({'font.size': 18,
                     # line widths
                     'lines.linewidth': 6,
                     })

import matplotlib.gridspec as gridspec
import subprocess
from torch import Tensor


def calc_hist(values, bins=100, density=True):
    """Utility: return (x, y) for a histogram."""
    v = values.detach().cpu().numpy().flatten()
    y, x = np.histogram(v, bins=bins, density=density)
    x = 0.5 * (x[:-1] + x[1:])
    return x, y


def plot_gaussians_params_histograms(
        data_groups: dict[str, list[Tensor]],
        psnrs,
        iters,
        out_path=CustomPath("dashboard.mp4"),
        max_frames=None,
        last_k_hist=5,
        save_last_time_only=False,
        save_video=False
):
    """
    Create a dashboard video visualizing parameter distributions and PSNR over iterations.
    Shows histograms of the last K iterations with color fading for comparison.
    """

    # if save video is true, save_last_time_only must be false
    assert not (save_video and save_last_time_only), "Cannot save video when save_last_time_only is True."

    # ---- Prepare parameter names ----
    sh_d = data_groups["shs"][0].shape[-1] // 3
    param_axis_names = {
        "opacities": [""],
        "means": ["x", "y", "z"],
        "scales": ["x", "y", "z"],
        "quats": ["x", "y", "z", "w"],
        "shs": [f"r{i}" for i in range(sh_d)]
               + [f"g{i}" for i in range(sh_d)]
               + [f"b{i}" for i in range(sh_d)],
    }

    # check shape of shs in first iteration
    # if data_groups["shs"][0].dim() == 3:
    #     g_shape = data_groups["shs"][0].shape
    #     reshaped_shs = []
    #     for iter_params in data_groups["shs"]:
    #         reshaped_shs.append(iter_params.reshape(-1, g_shape[1] * g_shape[2]))
    #     data_groups["shs"] = reshaped_shs

    # ---- Frame control ----
    T = len(iters)
    if max_frames is not None:
        T = min(T, max_frames)

    # ---- Prepare figure layout ----
    total_dims = sum(g[0].shape[-1] for g in data_groups.values())
    ncols = 5
    nrows = int(np.ceil(total_dims / ncols))

    fig = plt.figure(figsize=(5 * ncols, 3.5 * (nrows + 1)))
    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.5])
    axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]
    ax_psnr = fig.add_subplot(gs[-1, :])

    # ---- Precompute histograms and limits ----
    print("🔍 Precomputing histograms and axis limits...")
    subplot_map = []
    i = 0
    for key, iters_params in data_groups.items():

        D = iters_params.shape[-1]

        coord_names = [f"{key} {param_axis_names[key][d]}" for d in range(D)]

        for d in range(D):
            # g_at_t = [iters_params[t] for t in range(T)]
            all_hist_data = [calc_hist(iters_params[t][..., d], density=True) for t in range(T)]
            all_x, all_y = zip(*all_hist_data)
            xmin = min(x.min() for x in all_x)
            xmax = max(x.max() for x in all_x)
            # Center the x-axis around 0
            x_max_abs = max(abs(xmin), abs(xmax))
            xmin, xmax = -x_max_abs, x_max_abs
            ymin = 0.0
            ymax = max(y.max() for y in all_y) * 1.1
            subplot_map.append((key, d, axes[i], coord_names[d], all_x, all_y, xmin, xmax, ymin, ymax))
            i += 1

    # Hide unused subplots
    total_used_subplots = len(subplot_map)
    for j in range(total_used_subplots, len(axes)):
        axes[j].set_visible(False)

    # ---- Output folders ----
    out_dir = out_path.parent
    inter_dir = out_dir / "gaussians_histograms"
    inter_dir.mkdir(parents=True, exist_ok=True)

    print(f"📸 Generating histograms frames in {inter_dir:link}")

    # ---- Frame generation loop ----
    for frame_idx in range(T):

        if save_last_time_only and frame_idx < T - 1:
            continue

        fig.suptitle(f"Iteration {iters[frame_idx]} — PSNR: {psnrs[frame_idx]:.2f}", fontsize=18)

        for key, d, ax, name, all_x, all_y, xmin, xmax, ymin, ymax in subplot_map:
            ax.clear()

            # Plot last_k_hist iterations with progressive color fading
            k = min(last_k_hist, frame_idx + 1)
            idxs = list(range(frame_idx - k + 1, frame_idx + 1))
            for rel_i, hist_idx in enumerate(idxs):
                color = plt.cm.viridis(rel_i / max(1, k - 1))  # gradient color
                label = f"Iter {iters[hist_idx]}"
                ax.plot(all_x[hist_idx], all_y[hist_idx], color=color, alpha=0.9, lw=6, label=label)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(name)
            ax.legend(frameon=False, loc="upper right", fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)
            # Add vertical line at x=0 to show center
            ax.axvline(0, color='black', linewidth=1, linestyle=':', alpha=0.7)

        # ---- PSNR subplot ----
        ax_psnr.clear()
        ax_psnr.plot(iters[:frame_idx + 1], psnrs[:frame_idx + 1], color="#ffbc42", linewidth=8)
        ax_psnr.scatter(iters[frame_idx], psnrs[frame_idx], color="#ffbc42", s=60, zorder=3, linewidth=8)
        ax_psnr.set_xlim(min(iters), max(iters))
        ax_psnr.set_ylim(max(psnrs) * 0.7, max(psnrs) * 1.1)
        ax_psnr.set_title("PSNR Progress")
        ax_psnr.set_xlabel("Iteration")
        ax_psnr.set_ylabel("PSNR")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        frame_path = inter_dir / f"hist_{frame_idx:05d}.png"
        fig.savefig(frame_path, dpi=400)

    plt.close(fig)

    if not save_video:
        print(f"✅ Saved dashboard frames to {inter_dir} ({T} frames total)")
        return

    # ---- Combine with ffmpeg ----
    total_duration_sec = 20.0
    fps = T / total_duration_sec

    cmd = [
        "ffmpeg", "-y", "-framerate", f"{fps}",
        "-i", str(inter_dir / "hist_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", str(out_path)

    ]
    print("🎞️  Running FFmpeg to create video...")
    subprocess.run(cmd, check=True)

    print(f"✅ Saved dashboard video to {out_path} ({total_duration_sec:.1f}s total)")


def make_gaussians_dashboard_video_with_ani(data_groups, psnrs, iters, out_path=CustomPath("dashboard.mp4"),
                                            max_frames=None, scene=0):
    """
    Create a dashboard video visualizing parameter distributions and PSNR over iterations.
    Args:
        data_groups (dict): Dictionary containing parameter groups as keys and list of tensors as values.
                            Each list should have T entry of shape (N, D) where T is time
        psnrs (list): List of PSNR values over iterations.
        iters (list): List of iteration numbers corresponding to the PSNR values.
        out_path (CustomPath): Path to save the output video.
        max_frames (int, optional): Maximum number of frames to include in the video. If None, include all frames.
    """
    # Groups to visualize

    # Axis names for each parameter group
    # Calc sh axis names
    sh_d = data_groups["shs"][0][0].shape[-1] // 3
    param_axis_names = {
        "opacities": [""],
        "means": ["x", "y", "z"],
        "scales": ["x", "y", "z"],
        "quats": ["x", "y", "z", "w"],
        "shs": ["r" + str(i) for i in range(sh_d)] + [f"g{i}" for i in range(sh_d)] + [f"b{i}" for i in range(sh_d)],
    }

    T = list(data_groups.values())[0].shape[0]
    if max_frames is not None:
        T = min(T, max_frames)
    if iters is None:
        iters = list(range(T))

    # Count total subplots
    total_dims = sum(g.shape[-1] for g in data_groups.values())
    ncols = 4
    nrows = int(np.ceil(total_dims / ncols))

    # Use GridSpec to reserve one bottom row for PSNR plot
    fig = plt.figure(figsize=(5 * ncols, 3.5 * (nrows + 1)))
    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.5])
    axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]
    ax_psnr = fig.add_subplot(gs[-1, :])

    # Precompute all histograms and axis limits
    subplot_map = []
    i = 0
    for key, g in data_groups.items():
        D = g.shape[-1]
        coord_names = [f"{key} {param_axis_names[key][i]}" for i in range(D)]
        for d in range(D):
            all_hist_data = [calc_hist(g[t, scene, :, d], density=True) for t in range(T)]
            all_x, all_y = zip(*all_hist_data)
            # Find all time min/max for consistent axis limits
            xmin = min(x.min() for x in all_x)
            xmax = max(x.max() for x in all_x)
            ymin = 0.0
            ymax = max(y.max() for y in all_y) * 1.1
            subplot_map.append((key, d, axes[i], coord_names[d], all_x, all_y, xmin, xmax, ymin, ymax))
            i += 1

    # Animation update
    def update(frame_idx):
        fig.suptitle(f"Iteration {iters[frame_idx]} — PSNR: {psnrs[frame_idx]:.2f}", fontsize=18)

        for key, d, ax, name, all_x, all_y, xmin, xmax, ymin, ymax in subplot_map:
            ax.clear()
            ax.plot(all_x[frame_idx], all_y[frame_idx], color="#17becf", label=r"Resplat $\Delta$")

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(name)
            ax.legend(frameon=False, loc="upper left")

        # PSNR curve subplot
        ax_psnr.clear()
        ax_psnr.plot(iters[:frame_idx + 1], psnrs[:frame_idx + 1], color="#ffbc42")
        ax_psnr.scatter(iters[frame_idx], psnrs[frame_idx], color="#ffbc42", s=60, zorder=3)
        ax_psnr.set_xlim(min(iters), max(iters))
        ax_psnr.set_ylim(min(psnrs) * 0.98, max(psnrs) * 1.02)
        ax_psnr.set_title("PSNR Progress")
        ax_psnr.set_xlabel("Iteration")
        ax_psnr.set_ylabel("PSNR")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return axes + [ax_psnr]

    # Create video
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total_duration_sec = 20.0  # desired total duration
    interval_ms = total_duration_sec * 1000 / T  # milliseconds per frame

    try:
        ani = animation.FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
        ani.save(out_path, writer="ffmpeg", dpi=300)
    except FileNotFoundError:
        print("⚠️ FFmpeg not found. Saving as GIF instead.")
        ani = animation.FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
        ani.save(out_path.replace(".mp4", ".gif"), writer="pillow", dpi=300)

    plt.close(fig)
    print(f"✅ Saved dashboard video to {out_path} ({total_duration_sec:.1f}s total)")

    plt.close(fig)
    print(f"✅ Saved dashboard video to {out_path}")


# def make_dashboard_video(info, psnrs, iters, vanilla_lr, out_path="dashboard.mp4", max_frames=None):
#     # Groups to visualize
#     param_groups = ["opacities", "means", "scales", "rotations", "shs"]
#
#     # Axis names for each parameter group
#     # Calc sh axis names
#     sh_d = info["delta_shs"][0][0].shape[-1] // 3
#     param_axis_names = {
#         "opacities": [""],
#         "means": ["x", "y", "z"],
#         "scales": ["x", "y", "z"],
#         "rotations": ["x", "y", "z", "w"],
#         "shs": ["r" + str(i) for i in range(sh_d)] + [f"g{i}" for i in range(sh_d)] + [f"b{i}" for i in range(sh_d)],
#     }
#
#     # Extract and stack tensors
#     data = {}
#     for key in param_groups:
#         delta_data = torch.stack(info[f"delta_{key}"], dim=0)  # (T, B, N, D)
#         norm_grads_data = torch.stack(info[f"normalized_grad_{key}"], dim=0)  # (T, N, D)
#         data[key] = (delta_data, norm_grads_data)
#
#     T = list(data.values())[0][0].shape[0]
#     if max_frames is not None:
#         T = min(T, max_frames)
#     if iters is None:
#         iters = list(range(T))
#     scene = 0
#
#     # Compute axis limits for each param/dim
#     axis_limits = {}
#     for key, (delta_data, norm_grads_data) in data.items():
#         D = delta_data.shape[-1]
#         axis_limits[key] = []
#         for d in range(D):
#             delta_all = delta_data[:, scene, :, d].float().flatten().cpu().numpy()
#             grad_all = norm_grads_data[:, :, d].float().flatten().cpu().numpy() * vanilla_lr[key]
#             vmin = min(delta_all.min(), grad_all.min())
#             vmax = max(delta_all.max(), grad_all.max())
#
#             # Compute max y-density across all frames
#             y_max = 0.0
#             for t in range(T):
#                 delta = delta_data[t, scene, :, d].float().cpu().numpy()
#                 grad = norm_grads_data[t, :, d].float().cpu().numpy() * vanilla_lr[key]
#                 _, y1 = calc_hist(delta, density=True)
#                 _, y2 = calc_hist(grad, density=True)
#                 y_max = max(y_max, y1.max(), y2.max())
#
#             axis_limits[key].append((vmin, vmax, 0.0, y_max * 0.1))  # add small headroom
#
#     # Count total subplots
#     total_dims = sum(delta.shape[-1] for delta, _ in data.values())
#     ncols = 4
#     nrows = int(np.ceil(total_dims / ncols))
#
#     # Use GridSpec to reserve one bottom row for PSNR plot
#     fig = plt.figure(figsize=(5 * ncols, 3.5 * (nrows + 1)))
#     gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.5])
#     axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]
#     ax_psnr = fig.add_subplot(gs[-1, :])
#
#     subplot_map = []
#     i = 0
#     for key, (delta_data, norm_grads_data) in data.items():
#         D = delta_data.shape[-1]
#         coord_names = [f"{key} {param_axis_names[key][i]}" for i in range(D)]
#         for d in range(D):
#             subplot_map.append((key, d, axes[i], coord_names[d]))
#             i += 1
#
#     # Animation update
#     def update(frame_idx):
#         fig.suptitle(f"Iteration {iters[frame_idx]} — PSNR: {psnrs[frame_idx]:.2f}", fontsize=18)
#
#         for key, d, ax, name in subplot_map:
#             ax.clear()
#             delta_data, grads_data = data[key]
#             delta = delta_data[frame_idx, scene, :, d].float().cpu().numpy()
#             grad = grads_data[frame_idx, :, d].float().cpu().numpy() * vanilla_lr[key]
#             vmin, vmax, ymin, ymax = axis_limits[key][d]
#
#             # Δ histogram
#             x1, y1 = calc_hist(delta, density=True)
#             ax.plot(x1, y1, color="#17becf", label=r"Resplat $\Delta$")
#
#             # grad histogram
#             x2, y2 = calc_hist(grad, density=True)
#             ax.plot(x2, y2, color="#e377c2", ls="--", label=r"Adam $\Delta$")
#
#             ax.set_xlim(vmin, vmax)
#             ax.set_ylim(ymin, ymax)
#             ax.set_title(name)
#             ax.legend(frameon=False, loc="upper left")
#
#         # PSNR curve subplot
#         ax_psnr.clear()
#         ax_psnr.plot(iters[:frame_idx + 1], psnrs[:frame_idx + 1], color="#ffbc42")
#         ax_psnr.scatter(iters[frame_idx], psnrs[frame_idx], color="#ffbc42", s=60, zorder=3)
#         ax_psnr.set_xlim(min(iters), max(iters))
#         ax_psnr.set_ylim(min(psnrs) * 0.98, max(psnrs) * 1.02)
#         ax_psnr.set_title("PSNR Progress")
#         ax_psnr.set_xlabel("Iteration")
#         ax_psnr.set_ylabel("PSNR")
#
#         plt.tight_layout(rect=[0, 0, 1, 0.97])
#         return axes + [ax_psnr]
#
#     # Create video
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#
#     total_duration_sec = 20.0  # desired total duration
#     interval_ms = total_duration_sec * 1000 / T  # milliseconds per frame
#
#     try:
#         ani = animation.FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
#         ani.save(out_path, writer="ffmpeg", dpi=300)
#     except FileNotFoundError:
#         print("⚠️ FFmpeg not found. Saving as GIF instead.")
#         ani = animation.FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
#         ani.save(out_path.replace(".mp4", ".gif"), writer="pillow", dpi=300)
#
#     plt.close(fig)
#     print(f"✅ Saved dashboard video to {out_path} ({total_duration_sec:.1f}s total)")
#
#     plt.close(fig)
#     print(f"✅ Saved dashboard video to {out_path}")


import numpy as np
import torch
from pathlib import Path


def calc_hist(data, max_percentile=99.9, min_percentile=0.1, density=False):
    max_val = np.percentile(data, max_percentile)
    min_val = np.percentile(data, min_percentile)
    curr_data = data.clip(min_val, max_val)
    counts, bin_edges = np.histogram(curr_data, bins=100, range=(min_val, max_val), density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, counts


def debugging_convergence(
        deltas_list: list[dict[str, Tensor]],
        states_norms_list: list[Tensor],
        grads_raw_list: list[dict[str, Tensor]],
        normalized_grads_list: list[dict[str, Tensor]],
        psnr_list: list[float],
        iterations_list: list[int],
        output_path: Path,
        scene_name: str
):
    print("📈 Generating convergence plots...")
    assert len(iterations_list) > 0, "Iterations list cannot be empty."
    assert len(psnr_list) == len(iterations_list), "PSNR list length must match iterations list length."

    iters = iterations_list
    psnrs = psnr_list
    states_norms = []
    for state_norms in states_norms_list:
        states_norms.append(state_norms.mean().item())

    deltas_abs_means = []
    for deltas in deltas_list:
        total_mean = 0.0
        count = 0
        for key, delta in deltas.items():
            total_mean += delta.abs().mean().item()
            count += 1
        deltas_abs_means.append(total_mean / count)

    grads_raw_abs_means = []
    for grads in grads_raw_list:
        total_mean = 0.0
        count = 0
        for key, grad in grads.items():
            total_mean += grad.abs().mean().item()
            count += 1
        grads_raw_abs_means.append(total_mean / count)

    normalized_grads_abs_means = []
    for normalized_grads in normalized_grads_list:
        total_mean = 0.0
        count = 0
        for key, grad in normalized_grads.items():
            total_mean += grad.abs().mean().item()
            count += 1
        normalized_grads_abs_means.append(total_mean / count)

    # set rc once (inside context to avoid global mutation)
    rc = {
        'axes.titlesize': 17,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 11
    }

    with plt.rc_context(rc):
        # plot all quantities in one figure with 4 subplots
        fig, axs = plt.subplots(5, 1, figsize=(10, 15))
        # PSNR
        axs[0].plot(iters, psnrs, marker='o', color='blue')
        axs[0].set_title('PSNR over Iterations')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('PSNR')
        axs[0].grid(True, alpha=0.3)
        # State norm
        axs[1].plot(iters, states_norms, marker='o', color='orange')
        axs[1].set_title('State Norm over Iterations')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('State Norm')
        axs[1].grid(True, alpha=0.3)
        # Delta abs mean
        axs[2].plot(iters, deltas_abs_means, marker='o', color='green')
        axs[2].set_title('Mean Absolute Delta over Iterations')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Mean Absolute Delta')
        axs[2].grid(True, alpha=0.3)
        # Gradient abs mean
        axs[3].plot(iters, grads_raw_abs_means, marker='o', color='red', label='Raw Grads')
        axs[3].set_title('Mean Absolute Gradient over Iterations')
        axs[3].set_xlabel('Iteration')
        axs[3].set_ylabel('Mean Absolute Gradient')
        axs[3].grid(True, alpha=0.3)
        # Normalized Gradient abs mean
        axs[4].plot(iters, normalized_grads_abs_means, marker='o', color='purple', label='Normalized Grads')
        axs[4].set_title('Mean Absolute Normalized Gradient over Iterations')
        axs[4].set_xlabel('Iteration')
        axs[4].set_ylabel('Mean Absolute Normalized Gradient')
        axs[4].grid(True, alpha=0.3)
        plt.tight_layout()
        (output_path / "plots" / scene_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "plots" / scene_name / "convergence_plot.png", dpi=300)
        plt.close()


def debugging_deltas(
        deltas_list: list[dict[str, Tensor]],
        grads_list: list[dict[str, Tensor]],
        normalized_grads_list: list[dict[str, Tensor]],
        learning_rates: list[dict[str, float]],
        psnr_list: list[float],
        iterations_list: list[int],
        output_path: Path,
        scene_name: str
):
    assert len(iterations_list) > 0, "Iterations list cannot be empty."
    assert len(psnr_list) == len(iterations_list), "PSNR list length must match iterations list length."

    # Remove init.
    psnr_list = psnr_list[1:]
    iterations_list = iterations_list[1:]

    assert len(deltas_list) == len(iterations_list), "Deltas list length must match iterations list length."
    assert len(grads_list) == len(iterations_list), "Grads list length must match iterations list length."
    assert len(normalized_grads_list) == len(
        iterations_list), "Normalized grads list length must match iterations list length."
    if len(learning_rates) > 0:
        assert len(learning_rates) == len(
            iterations_list), "Learning rates list length must match iterations list length."
    iters = iterations_list
    psnrs = psnr_list
    # max_iter = max(iters) if len(iters) > 0 else 1
    nr_iters = len(iters)

    # set rc once (inside context to avoid global mutation)
    rc = {
        'axes.titlesize': 17,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 11
    }

    # Plot delta histograms
    for key in ["opacities", "means", "scales", "rotations"]:
        
        # TODO: log "sh0s", "shNs"

        # here N can change between iterations, C changes based on parameter type
        delta_data = [deltas[key] for deltas in deltas_list]  # list of [N, C]
        grads_data = [grads[key] for grads in grads_list]  # list of [N, C]
        normalized_grads_data = [normalized_grads[key] for normalized_grads in normalized_grads_list]  # list of [N, C]
        # if len(learning_rates) == 0:
        #     lr_data = [1.0] * len(delta_data)  # list of floats
        # else:
        #     lr_data = [lrs[key] for lrs in learning_rates]  # list of floats

        # Plot histogram of delta means for each step for each coordinate 
        D = delta_data[0].shape[-1]

        rows = 3  # delta, grad, normalized_grad

        with plt.rc_context(rc):
            plt.figure(figsize=(10 * D, 8 * rows))

            if D in [3, 4]:
                coord_names = ['X', 'Y', 'Z', 'W'][:D]
            elif D == 1:
                coord_names = [""]
            else:
                coord_names = [f"Dim {i}" for i in range(D)]

            for r, kind in enumerate(["delta", "grad", "grad_norm"]):
                for d in range(D):
                    ax = plt.subplot(rows, D, r * D + d + 1)

                    for i, t in enumerate(iters):
                        color_frac = float(i) / float(nr_iters)

                        # Select the correct dataset
                        if kind == "delta":
                            curr = delta_data[i][:, d].float().cpu().numpy()
                            cmap = plt.cm.viridis
                        elif kind == "grad":
                            curr = grads_data[i][:, d].float().cpu().numpy()
                            cmap = plt.cm.cividis
                        else:  # grad_norm
                            curr = normalized_grads_data[i][:, d].float().cpu().numpy()
                            cmap = plt.cm.plasma

                        # Compute histogram as normalized density
                        bin_centers, counts = calc_hist(curr)
                        max_counts = counts.max()
                        if max_counts > 0:
                            counts = counts / max_counts  # normalize peak=1

                        label = f"step: {t}, psnr: {psnrs[i]}"
                        ax.plot(bin_centers, counts, label=label,
                                color=cmap(color_frac), linewidth=2)

                    xlim = (-np.max(np.abs(bin_centers)), np.max(np.abs(bin_centers)))
                    ax.set_xlim(xlim)  # Center around 0
                    ax.axvline(0, color='black', linewidth=1, linestyle=':')  # vertical center line

                    if r == rows - 1:
                        ax.set_xlabel(f"{coord_names[d]}")
                    if d == 0:
                        ax.set_ylabel("Density")

                    # Titles
                    ax.set_title(f"{kind.replace('_', ' ').title()} {key.replace('_', ' ').title()} {coord_names[d]}")

                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

            plt.suptitle(f"{key.replace('_', ' ').title()} histograms (centered & normalized)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # save figure
            save_dir = os.path.join(output_path, "plots", scene_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{key}_deltas_histogram.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved delta histogram plot to {save_path}")

        # plt.figure(figsize=(10 * D, 10))
        # # Adjust font size
        # plt.rcParams.update({
        #     'axes.titlesize': 17,
        #     'axes.labelsize': 15,
        #     'xtick.labelsize': 15,
        #     'ytick.labelsize': 15,
        #     'legend.fontsize': 11  # Smaller
        # })
        # if D in [3, 4]:
        #     coord_names = ['X', 'Y', 'Z', 'W']
        # elif D == 1:
        #     coord_names = [""]
        # else:
        #     coord_names = [f"Dim {i}" for i in range(D)]

        # for d in range(D):
        #     plt.subplot(1, D, d + 1)
        #     for i, t in enumerate(iters):

        #         # Plot histogram of delta

        #         color = plt.cm.viridis(t / iters[-1])
        #         scene = 0
        #         curr_delta = delta_data[i, scene, :, d].float().cpu().numpy()
        #         bin_centers, counts = calc_hist(curr_delta)
        #         plt.plot(bin_centers, counts, label=fr"{psnrs[i]} $\Delta$ step {t}", color=color, linewidth=3)

        #         # Plot histogram of normalized grad

        #         color = plt.cm.plasma(t / iters[-1])
        #         curr_norm_grad = normalized_grads_data[i, :, d].float().cpu().numpy()
        #         bin_centers, counts = calc_hist(curr_norm_grad)
        #         plt.plot(bin_centers, counts, label=fr"{psnrs[i]} $g_t$ normalized step {t}", color=color,
        #                     linewidth=3,
        #                     linestyle='--')

        #     plt.xlabel(f"Delta {coord_names[d]}")
        #     plt.ylabel("Count")
        #     plt.title(f"{name} {coord_names[d]} histogram")

        #     # Arange irst delta handles and then normalized grad handles
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     delta_handles = [h for h, l in zip(handles, labels) if "Delta" in l]
        #     norm_grad_handles = [h for h, l in zip(handles, labels) if "g_t" in l]
        #     handles = delta_handles + norm_grad_handles
        #     labels = [l for l in labels if "Delta" in l] + [l for l in labels if "g_t" in l]
        #     plt.legend(handles, labels)
        # plt.tight_layout()

        # # save figure
        # save_path = output_path / "plots" / scene_name
        # os.makedirs(save_path, exist_ok=True)

        # save_path = save_path / f"{key}_deltas_histogram.png"
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved delta histogram plot to {save_path}")

    # # Plot delta cumsum
    # for key in ["delta_opacities"]:

    #     name = key.replace("_", " ").title()
    #     delta_data = deltas[key]  # list of [B, N, 3]
    #     delta_data = torch.stack(delta_data, dim=0)  # (steps, B, N, d)
    #     delta_cumsum = delta_data.cumsum(dim=0)  # (steps, B, N, d)

    #     # Plot cumsum of delta for randomly sampled 10 gaussians

    #     D = delta_data.shape[-1]
    #     plt.figure(figsize=(10 * D, 10))
    #     # Adjust font size
    #     plt.rcParams.update({
    #         'axes.titlesize': 17,
    #         'axes.labelsize': 15,
    #         'xtick.labelsize': 15,
    #         'ytick.labelsize': 15,
    #         'legend.fontsize': 11  # Smaller
    #     })

    #     indices = np.random.choice(delta_data.shape[2], size=20, replace=False, )
    #     scene = 0
    #     # get indices of the maximum cumsum at the last step
    #     indices = torch.argsort(delta_cumsum[-1, scene].abs().sum(dim=-1), descending=True)[:20].cpu().numpy()
    #     for d in range(D):
    #         plt.subplot(1, D, d + 1)
    #         for idx in indices:
    #             curr_delta = delta_cumsum[:, scene, idx, d].float().cpu().numpy()
    #             plt.plot(iters, curr_delta, label=f"Gaussian {idx}", linewidth=2)
    #         # plt.plot(iters, psnrs[1:], 'k--', label="PSNR", linewidth=4)

    #         plt.xlabel("Iteration")
    #         plt.ylabel(f"Accumulative of delta {name}")
    #         plt.title(f"Accumulative of delta {name} for 10 maximum gaussians")
    #         plt.grid(True)
    #         plt.legend()

    #     plt.tight_layout()
    #     # plt.show()

    #     raise NotImplementedError("Plot saving not implemented yet.")


# def debugging_reprojection_error(visualization_dump):
#     reprojection_error = visualization_dump['reprojection_error']  # list of list of (B, V, H*W, 2)
#     # Convert list of list to tensor
#     reprojection_error = [torch.stack(scene_errors, dim=0) for scene_errors in
#                             reprojection_error]  # list of (iterations, B, V, H*W, 2)
#     reprojection_error = torch.stack(reprojection_error, dim=0)  # (scenes, iterations, B, V, H*W, 2)
#     reprojection_error = torch.permute(reprojection_error,
#                                         (1, 0, 2, 3, 4, 5))  # [iterations, scenes, B, V, H*W, 2]
#     max_val = 3
#     reprojection_error = reprojection_error.clamp(0, max_val)
#     iterations = self.optimizer.save_every.get_iterations(len(reprojection_error))
#     target_psnrs = self.test_step_outputs_target["psnr"]  # list of psnr for target views per scene
#     target_psnrs = torch.Tensor(target_psnrs)  # [scenes, iterations]
#     target_psnrs = target_psnrs.mean(0)  # [iterations]

#     # Plot histograms of reprojection error through out the iterations

#     out_dir = self.test_cfg.output_path / "debugging"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     plt.figure(figsize=(6, 5))
#     for i, t in enumerate(iterations):
#         error = reprojection_error[i]
#         error_hist = error.view(-1).cpu().numpy()

#         counts, bin_edges = np.histogram(error_hist, bins=100, range=(0, max_val), density=False)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         plt.plot(bin_centers, counts,
#                     label=f"Iter {t}, psnr {target_psnrs[i]:.2f}",
#                     color=plt.cm.viridis(i / len(iterations)),
#                     linewidth=4,
#                     )

#         # plt.hist(error_hist, bins=100, range=(0, max_val), label=f"Iter {t}, psnr {target_psnrs[i]:.2f}",
#         #          histtype='step',
#         #          color=plt.cm.viridis(i / len(iterations)),
#         #          linewidth=4, )

#     # put the legend outside the plot to the right
#     plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
#     plt.xlabel("Reprojection error (pixels)")
#     plt.ylabel("Count")
#     plt.title("Reprojection error throughout test iterations")
#     plt.tight_layout()
#     # plt.show()
#     raise NotImplementedError("Saving reprojection error plots is not implemented yet.")

def debugging_gaussians(gaussian_list: list[Gaussians], psnr_list: list[float], iter_list: list[int], output_path: Path,
                        scene_name: str):
    assert len(gaussian_list) > 0, "Gaussian list cannot be empty."
    assert len(gaussian_list) == len(iter_list), "Gaussian list length must match iterations list length."
    assert len(psnr_list) == len(iter_list), "PSNR list length must match iterations list length."

    if gaussian_list[0].stores_activated:
        # need to invert the transformations
        scales_fn = torch.log
        opacities_fn = torch.logit
    else:
        # keep as is
        scales_fn = lambda x: x
        opacities_fn = lambda x: x

    # Extract gaussian attributes
    data_groups = {
        "opacities": [opacities_fn(g.opacities).squeeze(0).detach().cpu().unsqueeze(-1) for g in gaussian_list],
        "scales": [scales_fn(g.scales).squeeze(0).detach().cpu() for g in gaussian_list],
        "quats": [g.rotations.squeeze(0).detach().cpu() for g in gaussian_list],
        "means": [g.means.squeeze(0).detach().cpu() for g in gaussian_list],
        "shs": [g.harmonics.squeeze(0).detach().cpu() for g in gaussian_list]}

    plot_gaussians_params_histograms(
        data_groups=data_groups,
        psnrs=psnr_list,
        iters=iter_list,
        out_path=output_path / f"plots/{scene_name}/params.mp4"
    )


# def debugging_grads(visualization_dump):

#     # From post processing
#     gt = visualization_dump["grads"]  # list of list of list (Scenes, Steps, N, dim)
#     # Convert list of list to tensor
#     gt = [torch.stack(scene_grads, dim=0) for scene_grads in gt]  # list of (steps, N, dim)
#     gt = torch.stack(gt, dim=0)  # (scenes, steps, N, dim)

#     gt2 = gt ** 2

#     beta1 = 0.9
#     beta2 = 0.999
#     eps = 1e-8

#     # Calculate the moving averages of adam
#     mt = torch.zeros_like(gt)
#     vt = torch.zeros_like(gt)
#     mt2 = torch.zeros_like(gt2)
#     vt2 = torch.zeros_like(gt2)
#     mt_hat = torch.zeros_like(gt)
#     vt_hat = torch.zeros_like(gt)
#     for t in range(gt.shape[1]):
#         mt[:, t] = beta1 * mt[:, t - 1] + (1 - beta1) * gt[:, t] if t > 0 else (1 - beta1) * gt[:, t]
#         vt[:, t] = beta2 * vt[:, t - 1] + (1 - beta2) * gt[:, t] ** 2 if t > 0 else (1 - beta2) * gt[:, t] ** 2
#         mt2[:, t] = beta1 * mt2[:, t - 1] + (1 - beta1) * gt2[:, t] if t > 0 else (1 - beta1) * gt2[:, t]
#         vt2[:, t] = beta2 * vt2[:, t - 1] + (1 - beta2) * gt2[:, t] ** 2 if t > 0 else (1 - beta2) * gt2[:, t] ** 2
#         mt_hat[:, t] = mt[:, t] / (1 - beta1 ** (t + 1))
#         vt_hat[:, t] = vt[:, t] / (1 - beta2 ** (t + 1))

#     denom = torch.sqrt(vt_hat) + eps
#     delta = mt_hat / denom

#     # Plot histograms of gt, gt^2, mt_hat, vt_hat, delta

#     # Adjust font size
#     plt.rcParams.update({
#         'axes.titlesize': 17,
#         'axes.labelsize': 15,
#         'xtick.labelsize': 15,
#         'ytick.labelsize': 15,
#         'legend.fontsize': 9  # Smaller
#     })
#     d = 0  # means x
#     d = 2  # means z
#     scene = 0
#     plt.figure(figsize=(20, 15))
#     names = [r"$g_t$", r"$g_t^2$", r"$\hat{m}_t$", r"$\hat{v}_t$", r"$\sqrt{\hat{v}_t} + \epsilon$", r"$\Delta$"]
#     data_list = [gt, gt2, mt_hat, vt_hat, denom, delta]
#     for i, (name, data) in enumerate(zip(names, data_list)):
#         plt.subplot(2, 3, i + 1)

#         T = data.shape[1]
#         for t in range(T):
#             data_t = data[scene, t, :, d]
#             max_val = np.percentile(data_t, 99.9)
#             min_val = np.percentile(data_t, 0.1)
#             data_t = data_t.clamp(min_val, max_val).cpu().numpy()
#             counts, bin_edges = np.histogram(data_t, bins=100, range=(min_val, max_val), density=False)
#             bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#             # plot color with virdis colormap
#             color = plt.cm.viridis(t / T)
#             plt.plot(bin_centers, counts, label=fr"step {t}", color=color, linewidth=3)
#             # plt.xlim((min_val, max_val))
#         plt.xlabel(name)
#         plt.ylabel("Count")
#         plt.title(f"{name}")
#         plt.legend()
#     plt.suptitle(f"Histograms of Adam statistics for gradient element {d}")
#     plt.tight_layout()
#     # plt.show()
#     raise NotImplementedError("Plot saving not implemented yet.")

#     # Compare gt to mt
#     gt_mt_diff = gt - mt_hat
#     plt.figure(figsize=(6, 5))
#     T = gt_mt_diff.shape[1]
#     for t in range(1, T):
#         data = gt_mt_diff[scene, t, :, d]
#         max_val = np.percentile(data, 99.9)
#         min_val = np.percentile(data, 0.1)
#         data = data.clamp(min=min_val, max=max_val)
#         counts, bin_edges = np.histogram(data, bins=100, range=(min_val, max_val), density=False)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         # plot color with virdis colormap
#         color = plt.cm.viridis(t / T)
#         plt.plot(bin_centers, counts, label=fr"step {t}", color=color, linewidth=3)
#     plt.xlabel(r"$g_t - \hat{m}_t$")
#     plt.ylabel("Count")
#     plt.title(r"Histogram of $g_t - \hat{m}_t$")
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(True)
#     # plt.show()
#     raise NotImplementedError("Plot saving not implemented yet.")

#     # Compaer |gt| to sqrt(vt) + eps
#     gt_abs_ratio = gt.abs() / denom
#     plt.figure(figsize=(6, 5))
#     T = gt_abs_ratio.shape[1]
#     for t in range(1, T):
#         data = gt_abs_ratio[scene, t, :, d]
#         max_val = np.percentile(data, 99.9)
#         min_val = np.percentile(data, 0.1)
#         data = data.clamp(min=min_val, max=max_val)
#         counts, bin_edges = np.histogram(data, bins=100, range=(min_val, max_val), density=False)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         # plot color with virdis colormap
#         color = plt.cm.viridis(t / T)
#         plt.plot(bin_centers, counts, label=fr"step {t}", color=color, linewidth=3)
#     plt.xlabel(r"$|g_t| / (\sqrt{\hat{v}_t} + \epsilon)$")
#     plt.ylabel("Count")
#     plt.title(r"Histogram of $|g_t| / (\sqrt{\hat{v}_t} + \epsilon)$")
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(True)
#     # plt.show()
#     raise NotImplementedError("Plot saving not implemented yet.")

#     # Compare gt to delta
#     delta_ratio = delta / gt
#     plt.figure(figsize=(10, 5))
#     T = delta_ratio.shape[1]
#     for t in range(1, T):
#         data = delta_ratio[scene, t, :, d]
#         max_val = np.percentile(data, 99.9)
#         min_val = np.percentile(data, 0.1)
#         data = data.clamp(min=min_val, max=max_val)
#         counts, bin_edges = np.histogram(data, bins=100, range=(min_val, max_val), density=False)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         # plot color with virdis colormap
#         color = plt.cm.viridis(t / T)
#         plt.plot(bin_centers, counts, label=fr"step {t}", color=color, linewidth=3)
#     plt.xlabel(r"$g_t / \Delta$")
#     plt.ylabel("Count")
#     plt.title(r"Histogram of $g_t / \Delta$")
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(True)
#     # plt.show()
#     raise NotImplementedError("Plot saving not implemented yet.")

#     # Plot gaussian postion in 2d
#     i = 10000  # gaussian index
#     scene = 0
#     grads_xy = gt[..., :2]  # (scenes, steps, N, 2)
#     deltas_xy = gt[..., 2:]
#     gt_xy_pos = grads_xy.cumsum(dim=2)  # cumulative sum to get positions
#     deltas_xy_pos = deltas_xy.cumsum(dim=2)  # cumulative sum to get positions

#     # Plot different gaussian position
#     plt.figure(figsize=(6, 6))
#     gaussian_pos = gt_xy_pos[scene, :, i, :]  # (steps, 2)
#     # Plot with color gradient from blue to red
#     plt.scatter(gaussian_pos[:, 0].cpu(), gaussian_pos[:, 1].cpu(), c=np.linspace(0, 1, len(gaussian_pos)),
#                 cmap='viridis')
#     # plt.plot(gaussian_pos[:, 0].cpu(), gaussian_pos[:, 1].cpu(), marker='o', colors=)
#     plt.scatter(gaussian_pos[0, 0].cpu(), gaussian_pos[0, 1].cpu(), color='green', label='Start', s=100)
#     plt.scatter(gaussian_pos[-1, 0].cpu(), gaussian_pos[-1, 1].cpu(), color='red', label='End', s=100)
#     plt.title(f"Gaussian {i} position through steps (from green to red)")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.axis('equal')
#     plt.grid(True)
#     plt.legend()
#     # plt.show()
#     raise NotImplementedError("Plot saving not implemented yet.")


def debugging_invisible_gaussians(
        gaussian_list,
        grads_raw_list,
        normalized_grads_list,
        means2d_list,
        radii_list,
        psnr_list,
        iterations_list,
        output_path,
        scene_name
):
    def concat_grads(grads_list):
        grads_per_params = []
        G = grads_list[0][list(grads_list[0].keys())[0]].shape[0]  # number of gaussians
        for key in grads_list[0].keys():
            grads_val = [grads[key].reshape(G, -1) for grads in grads_list]

            grads_per_params.append(torch.stack(grads_val, dim=0))  # (T, G, D)

        grads_mat = torch.cat(grads_per_params, dim=-1)  # (T, G, D)
        return grads_mat, grads_per_params

    # === Prepare data ===
    grads_mat, grads_per_params = concat_grads(grads_raw_list)  # (T, G, D)
    norm_grads_mat, norm_grads_per_params = concat_grads(normalized_grads_list)  # (T, G, D)
    scales_grads = grads_per_params[1]  # (T, G, scale_dim)
    opacities_grads = grads_per_params[3]  # (T, G, opacity_dim)
    scales_norm_grads = norm_grads_per_params[1]  # (T, G, scale_dim)
    opacities_norm_grads = norm_grads_per_params[3]  # (T, G, opacity_dim)
    means2d = torch.cat(means2d_list, dim=0).cpu()[1:]  # (T, V, G, 2)
    radii_list = torch.cat(means2d_list, dim=0).cpu()[1:]

    T, G, D = grads_mat.shape
    iterations_list = iterations_list[1:]  # remove init.

    # === Convert Gaussian params to tensor ===
    def extract_params(gaussians: list[Gaussians], grads):
        params = []
        for k in grads[0].keys():
            if k in ["shNs", "sh0s"]:
                continue
            params.append(torch.stack([getattr(g, k)[0].detach().cpu() for g in gaussians]))
        params.append(torch.stack([g.harmonics[0].detach().cpu() for g in gaussians]))
        params = [p[1:] for p in params]  # remove init., each (T, G, dim)
        gaussians_mat = torch.cat([p.reshape(T, G, -1) for p in params], dim=-1)  # (T, G, D)
        return params, gaussians_mat

    params_mat, gaussians_mat = extract_params(gaussian_list, grads_raw_list)  # (T, G, D)
    means = params_mat[0]
    scales = params_mat[1]
    rotations = params_mat[2]
    opacities = params_mat[3]
    harmonics = params_mat[4]

    # === Compute zero / partial grad masks ===
    zero_grad_mask = (grads_mat == 0)  # (T, G, D)
    zero_grad_cnt = (zero_grad_mask).sum(dim=-1)  # (T, G)
    is_zero = (zero_grad_mask).all(dim=-1)  # (T, G)
    is_nonzero = (~zero_grad_mask).all(dim=-1)  # (T, G)
    is_partial = ~(is_zero | is_nonzero)  # (T, G)
    validation = is_zero.float() + is_nonzero.float() + is_partial.float()
    assert (validation == 1).all(), "Gradient classification error: some Gaussians are not classified properly."

    # 0 = zero, 1 = partial, 2 = nonzero
    state = torch.zeros_like(is_zero, dtype=torch.int8)
    state[is_partial] = 1
    state[is_nonzero] = 2

    # === Compute change in zero grad masks ===
    transition = state[1:] - state[:-1]  # (T-1, G)
    transition_per_gaussian = (transition != 0).sum(dim=0)  # (G,)

    # === Compute counts ===
    zero_cnt = is_zero.sum(dim=1).cpu().numpy()  # (T,)
    partial_cnt = is_partial.sum(dim=1).cpu().numpy()  # (T,)

    # === Compute change in zero grad masks ===
    zero_to_partial = ((state[:-1] == 0) & (state[1:] == 1)).sum(dim=1)
    zero_to_nonzero = ((state[:-1] == 0) & (state[1:] == 2)).sum(dim=1)
    partial_to_nonzero = ((state[:-1] == 1) & (state[1:] == 2)).sum(dim=1)
    partial_to_zero = ((state[:-1] == 1) & (state[1:] == 0)).sum(dim=1)
    nonzero_to_zero = ((state[:-1] == 2) & (state[1:] == 0)).sum(dim=1)
    nonzero_to_partial = ((state[:-1] == 2) & (state[1:] == 1)).sum(dim=1)

    # Stay as is
    zero_to_zero = ((state[:-1] == 0) & (state[1:] == 0)).sum(dim=1)
    partial_to_partial = ((state[:-1] == 1) & (state[1:] == 1)).sum(dim=1)
    nonzero_to_nonzero = ((state[:-1] == 2) & (state[1:] == 2)).sum(dim=1)

    total = (zero_to_nonzero + zero_to_partial + partial_to_nonzero + partial_to_zero + nonzero_to_zero
             + nonzero_to_partial + zero_to_zero + partial_to_partial + nonzero_to_nonzero)
    assert (total == G).all(), "Transition counts do not sum up to total number"

    # ===  Gaussian indices ===
    n_vis = 30
    # random_mask = ((state[:-1] == 0) & (state[1:] == 0))
    # random_indices = torch.where(random_mask)
    # random_indices = random_indices[1].unique()
    # random_indices = random_indices[torch.randperm(len(random_indices))[:n_vis]]

    # Extract indices of the largest scale gaussians
    top_scales = torch.topk(scales[-1, ..., 0], k=n_vis, largest=True).indices
    random_indices = top_scales

    # === Compute mean param/grad time series ===
    # Zero-grad & partial-grad subsets are time-varying masks.
    grad_norms = grads_mat.norm(dim=-1)  # (T, G)

    # === Create figure ===
    fig, axes = plt.subplots(10, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"Debugging Invisible Gaussians — {scene_name}", fontsize=16)

    # 1️⃣ Zero-grad count
    i = 0
    axes[i].plot(iterations_list, zero_cnt, label="Zero Grad Gaussians")
    axes[i].plot(iterations_list, partial_cnt, label="Partial Grad Gaussians")
    axes[i].set_ylabel("Count")
    axes[i].set_title("Zero vs Partial Grad Gaussians Count")
    axes[i].legend()

    # Change of classification counts
    i += 1
    axes[i].plot(iterations_list[1:], zero_to_partial.cpu(), label='Zero → Partial')
    axes[i].plot(iterations_list[1:], zero_to_nonzero.cpu(), label='Zero → Nonzero')
    axes[i].plot(iterations_list[1:], partial_to_nonzero.cpu(), label='Partial → Nonzero')
    axes[i].plot(iterations_list[1:], partial_to_zero.cpu(), label='Partial → Zero')
    axes[i].plot(iterations_list[1:], nonzero_to_zero.cpu(), label='Nonzero → Zero')
    axes[i].plot(iterations_list[1:], nonzero_to_partial.cpu(), label='Nonzero → Partial')
    axes[i].set_ylabel("Count")
    axes[i].set_title("Transition Grad Gaussians Count")
    axes[i].legend()

    # 2️⃣ Random zero grad cnt
    i += 1
    axes[i].plot(iterations_list, zero_grad_cnt[:, random_indices])
    axes[i].set_title("Gaussians zero grad count")
    axes[i].set_ylabel("Zero grad cnt")

    # 3️⃣ Random Gradient magnitudes
    i += 1
    axes[i].plot(iterations_list, grad_norms[:, random_indices])
    axes[i].set_title("Gaussians gradient magnitude")
    axes[i].set_ylabel("Gradient norm")

    # 4️⃣ Random scales
    i += 1
    axes[i].plot(iterations_list, scales[:, random_indices].mean(-1))
    axes[i].set_title("Gaussians scales")
    axes[i].set_ylabel("Scales")

    # 4️⃣ Random opacities
    i += 1
    axes[i].plot(iterations_list, opacities[:, random_indices])
    axes[i].set_title("Gaussians opacities")
    axes[i].set_ylabel("Opacities")

    # 4️⃣ Random means
    i += 1
    axes[i].plot(iterations_list, scales_norm_grads[:, random_indices, 0])
    axes[i].set_title("Gaussians scales X adam grad")
    axes[i].set_ylabel("Scales X grad")

    i += 1
    axes[i].plot(iterations_list, opacities_grads[:, random_indices, 0])
    axes[i].set_title("Gaussians opacities adam grad")
    axes[i].set_ylabel("Opacities X grad")

    i += 1
    axes[i].plot(iterations_list, means2d[:, 0, random_indices, 0])
    axes[i].set_title("Gaussians means 2D X")
    axes[i].set_ylabel("Means 2D X")
    axes[i].set_xlabel("Iteration")

    i += 1
    axes[i].plot(iterations_list, radii_list[:, :, random_indices, 0].sum(1))
    axes[i].set_title("Gaussians radii 2D X")
    axes[i].set_ylabel("Radii 2D X")
    axes[i].set_xlabel("Iteration")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # === Save figure ===
    # os.makedirs(output_path, exist_ok=True)
    # fig_path = os.path.join(output_path, f"{scene_name}_debug_invisible_gaussians_over_time.png")
    # plt.savefig(fig_path)
    # plt.close(fig)
    plt.show()

    print(f"✅ Saved time-evolution debug plot → {fig_path}")
