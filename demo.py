"""End-to-end OptGS demo on a COLMAP scene.

Main-codebase port of ``baselines/gsplat/examples/simple_trainer_optgs.py``:
same flow — SfM-initialize Gaussians, refine them with the learned optimizer
via the ``OptGS`` API, evaluate on held-out views — but using only the
``optgs`` package (no gsplat / gsplat-examples dependency):

    from optgs.experimental.api import OptGS

    optgs = OptGS(checkpoint="hf://org/repo/model.ckpt", device="cuda")
    optgs.initialize_from_tensors(gaussians, batched_views)
    refined = optgs.optimize()          # learned optimization

COLMAP loading uses ``optgs.dataset.colmap``; the SfM init builds an optgs
``Gaussians`` directly via ``points_to_gaussians``; evaluation renders with
the optimizer's own decoder.

The scene is refined three ways and compared on held-out views: the learned
optimizer (Learn2Splat) with the *dense* and the *sparse* checkpoint, and a
3DGS Adam baseline (gsplat hyperparameters). All run through the same
``optimize()`` path with identical SfM init, view minibatches and step budget.
Each uses its checkpoint's gsplat renderer; ``--rasterize-mode`` / ``--eps2d``
pin one renderer across all runs.

Usage (run from the repo root, with ``optgs`` importable):

    python demo.py                    # headless: dense + sparse checkpoints + an Adam baseline
    python demo.py --with-gui server  # interactive viser GUI (frames rendered by the decoder)
    python demo.py --with-gui client  # interactive viser GUI (viser's WebGL splat renderer)
    python demo.py --with-gui gradio  # interactive gradio GUI (streamed renders + Model3D splats)

The demo scene and the checkpoints are fetched from the Hugging Face Hub on
first run (cached under ./data and ./checkpoints). A CUDA device is required.
"""

import warnings

# Demo: silence third-party UserWarnings (xFormers/flash-attn not installed,
# Hydra's _self_ notice, pointops' deprecated tensor constructors) for clean output.
warnings.filterwarnings("ignore")

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from rich.console import Console
from rich.table import Table
from torch import Tensor

console = Console()

from optgs.dataset.colmap.utils import Dataset, Parser
from optgs.experimental.initializers_utils import knn, points_to_gaussians
from optgs.model.types import Gaussians
from optgs.scene_trainer.common.gaussian_adapter import build_covariance

# Camera near/far planes — inria's znear/zfar (also the optgs colmap-dataset
# constants). Fixed; not a user knob.
NEAR_PLANE = 0.01
FAR_PLANE = 100.0

# Spherical-harmonics DC -> RGB (3DGS convention: rgb = 0.5 + C0 * dc). Colours
# the splats for viser's client-side renderer.
SH_C0 = 0.28209479177387814

# The demo scene is fetched from this Hugging Face repo on first run. The repo
# mirrors the local layout, so e.g. ``data/mip360/garden`` in the repo lands at
# ``./data/mip360/garden``.
DEMO_DATA_REPO = "autonomousvision/learn2splat"

# Learned-optimizer checkpoints on the Hugging Face Hub. hf:// refs are fetched
# and cached under ./checkpoints on first use (see optgs.misc.hf_ckpt).
CHECKPOINTS = {
    "dense": "hf://autonomousvision/learn2splat/dense/checkpoints/epoch_5-step_50000.ckpt",
    "sparse": "hf://autonomousvision/learn2splat/sparse/checkpoints/epoch_9-step_90000.ckpt",
}


def ensure_data(data_dir: str) -> None:
    """Download the demo scene from the Hugging Face Hub if it is not present."""
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        return
    from huggingface_hub import snapshot_download

    console.print(
        f"[yellow]{data_dir}[/] not found — downloading from "
        f"[cyan]hf://{DEMO_DATA_REPO}[/] …"
    )
    snapshot_download(
        repo_id=DEMO_DATA_REPO,
        allow_patterns=[f"{data_dir.rstrip('/')}/**"],
        local_dir=".",
    )
    console.print(f"[green]✓[/] scene ready at [yellow]{data_dir}[/]")


@dataclass
class Config:
    # Path to the COLMAP dataset (expects images/ + sparse/0/).
    data_dir: str = "data/mip360/garden"
    # Downsample factor for the dataset.
    data_factor: int = 4
    # Global multiplier on scene-size-related parameters.
    global_scale: float = 1.0
    # Normalize the world space.
    normalize_world_space: bool = True
    # Every N images is a test image, held out for evaluation.
    test_every: int = 8
    # Directory to save renders / stats / the refined PLY.
    result_dir: str = "results/demo"
    # Random seed.
    seed: int = 42

    # --- Interactive GUI ---
    # Launch an interactive GUI instead of the headless comparison. viser:
    # "server" renders frames with the optgs decoder, "client" uses viser's
    # built-in WebGL Gaussian-splat renderer. "gradio" runs a browser GUI
    # (decoder renders streamed live + an interactive Model3D splat viewer for
    # the result). Unset = headless run.
    with_gui: Optional[Literal["client", "server", "gradio"]] = None
    # Port for the GUI web server (--with-gui only).
    gui_port: int = 8080

    # --- OptGS learned optimizer ---
    # Compute device (OptGS requires CUDA).
    device: str = "cuda"
    # Number of learned refinement steps.
    max_steps: int = 100
    # Views the optimizer sees per refinement step (the view minibatch).
    opt_batch_size: int = 8
    # View-minibatch sampling strategy: "random", "sequential", or "fps"
    # (farthest-point sampling over camera positions).
    opt_batch_strategy: Literal["random", "sequential", "fps"] = "fps"

    # --- gsplat renderer ---
    # rasterize_mode / eps2d: when set, applied to every run (dense, sparse,
    # Adam), overriding each checkpoint's decoder config so the comparison uses
    # one renderer. Left unset, each run uses its own checkpoint's value.
    rasterize_mode: Optional[Literal["classic", "antialiased"]] = None
    eps2d: Optional[float] = None

    # --- Initialization ---
    # Initialization strategy: "sfm" or "random".
    init_type: str = "sfm"
    # Initial number of GSs. Ignored when init_type="sfm".
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the scene extent (random init).
    init_extent: float = 3.0
    # Initial opacity / scale of each GS.
    init_opa: float = 0.1
    init_scale: float = 1.0


def scene_extent(parser: Parser, global_scale: float) -> float:
    """Scene-size scalar: parser extent x 1.1 x global_scale."""
    return parser.scene_scale * 1.1 * global_scale


def sfm_initialization(
    parser: Parser, cfg: Config, sh_degree: int, device: torch.device, dtype: torch.dtype
) -> Gaussians:
    """SfM (or random) Gaussian init -> an optgs ``Gaussians`` (batch=1).

    Builds the parameter tensors with the same heuristics as 3DGS / the optgs
    COLMAP initializer, then assembles them through ``points_to_gaussians``.
    """
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif cfg.init_type == "random":
        extent = scene_extent(parser, cfg.global_scale)
        points = cfg.init_extent * extent * (
            torch.rand((cfg.init_num_pts, 3)) * 2 - 1
        )
        rgbs = torch.rand((cfg.init_num_pts, 3))
    else:
        raise ValueError(f"unknown init_type: {cfg.init_type!r} (sfm | random)")

    # GS size = average distance to the 3 nearest neighbours ([:, 1:] drops self).
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    scales = (torch.sqrt(dist2_avg) * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    opacities = torch.full((points.shape[0],), cfg.init_opa)

    # points_to_gaussians returns pre-activation params (log scales, logit
    # opacity, sh0/shN, random quats).
    g = points_to_gaussians(
        {"xyz": points, "rgb": rgbs, "scales": scales, "opacities": opacities},
        sh_degree=sh_degree,
        device=device,
    )
    sh0, shN = g["sh0"], g["shN"]
    harmonics = torch.cat([sh0, shN], dim=1) if shN is not None else sh0  # [N, K, 3]
    harmonics = harmonics.permute(0, 2, 1)  # -> [N, 3, K]

    scales_act = torch.exp(g["scales_raw"])
    opacities_act = torch.sigmoid(g["opacities_raw"])
    rotations = F.normalize(g["rotations_unnorm"], dim=-1)
    covariances = build_covariance(scale=scales_act, rotation_xyzw=rotations)

    def _b(t: Tensor) -> Tensor:  # add the batch dimension and cast
        return t.unsqueeze(0).to(dtype)

    return Gaussians(
        means=_b(g["xyz"]),
        covariances=_b(covariances),
        harmonics=_b(harmonics),
        opacities=_b(opacities_act),
        scales=_b(scales_act),
        rotations=_b(rotations),
        rotations_unnorm=_b(g["rotations_unnorm"]),
    )


def collect_cameras(
    dataset: Dataset, indices: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Stack the selected views into ``(camtoworlds, Ks, images)``.

    ``images`` is returned in [0, 1]. All views must share one (H, W) — the
    optgs renderer takes a single image shape.
    """
    c2ws, ks, imgs = [], [], []
    hw = None
    for i in indices:
        data = dataset[i]
        img = data["image"] / 255.0  # [H, W, 3], float
        if hw is None:
            hw = img.shape[:2]
        elif img.shape[:2] != hw:
            raise ValueError(
                f"all views must share one (H, W); got {tuple(img.shape[:2])} "
                f"vs {tuple(hw)}. Render the dataset at a single resolution."
            )
        c2ws.append(data["camtoworld"])
        ks.append(data["K"])
        imgs.append(img)
    return torch.stack(c2ws), torch.stack(ks), torch.stack(imgs)


def build_batched_views(
    camtoworlds: Tensor,
    Ks: Tensor,
    images: Tensor,
    scene_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """COLMAP cameras -> an optgs ``BatchedViews`` dict (batch=1).

    COLMAP ``camtoworld`` is already optgs's extrinsics convention (OpenCV
    camera->world). ``K`` is pixel-space; optgs wants it normalized by image
    width/height.
    """
    v, h, w = images.shape[0], images.shape[1], images.shape[2]

    Ks_norm = Ks.clone()
    Ks_norm[:, 0, :] /= w  # normalized focal / principal point
    Ks_norm[:, 1, :] /= h

    image = images.permute(0, 3, 1, 2)  # [V, 3, H, W]

    def _b(t: Tensor) -> Tensor:  # add the batch dimension and move to device
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    return {
        "extrinsics": _b(camtoworlds),
        "intrinsics": _b(Ks_norm),
        "image": _b(image),
        "near": torch.full((1, v), NEAR_PLANE, device=device, dtype=dtype),
        "far": torch.full((1, v), FAR_PLANE, device=device, dtype=dtype),
        "index": torch.arange(v, device=device).unsqueeze(0),
        "scene_scale": torch.tensor([scene_scale], device=device, dtype=dtype),
    }


@torch.no_grad()
def render_and_score(
    optgs,
    refined: Gaussians,
    val_bv: dict,
    val_images: Tensor,
    out_dir: str,
    device: torch.device,
) -> dict:
    """Render one optimizer's result on the held-out views; report mean PSNR.

    Saves a ``gt | pred`` strip per view under ``out_dir/renders``.
    """
    render_dir = os.path.join(out_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    h, w = val_images.shape[1], val_images.shape[2]

    out = optgs.decoder.forward(
        refined, val_bv["extrinsics"], val_bv["intrinsics"],
        val_bv["near"], val_bv["far"], image_shape=(h, w),
    )
    colors = out.color[0].clamp(0.0, 1.0)  # [V, 3, H, W]

    psnrs = []
    for i in range(colors.shape[0]):
        gt = val_images[i].to(device)  # [H, W, 3]
        pred = colors[i].permute(1, 2, 0)
        psnrs.append(-10.0 * torch.log10(torch.mean((pred - gt) ** 2)).item())

        canvas = torch.cat([gt, pred], dim=1).cpu().numpy()  # gt | pred
        imageio.imwrite(
            os.path.join(render_dir, f"val_{i:04d}.png"),
            (canvas * 255).astype(np.uint8),
        )

    return {"psnr": float(np.mean(psnrs)), "num_views": int(colors.shape[0])}


@torch.no_grad()
def render_view(
    optgs, gaussians: Gaussians, camera, height: int,
    device: torch.device, dtype: torch.dtype,
) -> np.ndarray:
    """Render ``gaussians`` from a viser camera into an ``[H, W, 3]`` uint8 image.

    viser cameras follow OpenCV conventions, so ``(wxyz, position)`` is directly
    the camera-to-world transform the optgs decoder expects — no axis flip.
    """
    import viser.transforms as vtf

    from optgs.misc.image_io import prep_image

    h = int(height)
    w = max(1, round(h * camera.aspect))  # camera.aspect = width / height

    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :3] = torch.tensor(
        vtf.SO3(camera.wxyz).as_matrix(), device=device, dtype=dtype
    )
    c2w[:3, 3] = torch.tensor(camera.position, device=device, dtype=dtype)

    # Normalized intrinsics from the vertical fov; the decoder un-normalizes by
    # the image width/height.
    fy = (h / 2.0) / float(np.tan(camera.fov / 2.0))
    K = torch.eye(3, device=device, dtype=dtype)
    K[0, 0] = fy / w
    K[1, 1] = fy / h
    K[0, 2] = 0.5
    K[1, 2] = 0.5

    near = torch.full((1, 1), NEAR_PLANE, device=device, dtype=dtype)
    far = torch.full((1, 1), FAR_PLANE, device=device, dtype=dtype)
    out = optgs.decoder.forward(
        gaussians, c2w[None, None], K[None, None], near, far, image_shape=(h, w),
    )
    return prep_image(out.color[0, 0])  # [H, W, 3] uint8


def gaussians_to_splat_data(gaussians: Gaussians) -> dict:
    """An optgs ``Gaussians`` (batch=1) -> numpy arrays for viser's splat viewer.

    Covariances are recomputed from scale/rotation (the optimizer updates those
    but may leave the optional ``Gaussians.covariances`` field stale); colours
    come from the SH DC term (degree 0 — viser's renderer is not view-dependent).
    """
    scales = gaussians.scales[0]
    opacities = gaussians.opacities[0]
    if not gaussians.stores_activated:
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)
    rotations = F.normalize(gaussians.rotations_unnorm[0], dim=-1)
    covariances = build_covariance(scale=scales, rotation_xyzw=rotations)
    rgbs = (0.5 + SH_C0 * gaussians.harmonics[0, :, :, 0]).clamp(0.0, 1.0)

    def _np(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    return {
        "centers": _np(gaussians.means[0]),          # (N, 3)
        "covariances": _np(covariances),             # (N, 3, 3)
        "rgbs": _np(rgbs),                           # (N, 3)
        "opacities": _np(opacities.reshape(-1, 1)),  # (N, 1)
    }


def run_gui(
    instances: dict,
    gaussians: Gaussians,
    train_bv: dict,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Interactive viser GUI: watch the optimization, pick an optimizer, reset.

    The initialization is shown first; the user picks an optimizer — the
    Learn2Splat learned optimizer (dense or sparse checkpoint) or a 3DGS Adam
    baseline — and clicks Start; every optimizer step is rendered and displayed;
    Reset restores the initialization. ``cfg.with_gui`` chooses the renderer —
    "server" (optgs decoder, frames streamed as images) or "client" (viser's
    WebGL splats).

    ``instances`` maps "dense"/"sparse" to their initialized ``OptGS``.
    """
    import threading

    import viser
    import viser.transforms as vtf

    from optgs.experimental.api.integration.config_bridge import build_adam_baseline

    mode = cfg.with_gui  # "server" | "client"
    server = viser.ViserServer(port=cfg.gui_port)

    # Optimizer dropdown label -> (instances key, whether to swap in Adam).
    # "dense"/"sparse" run that checkpoint's own learned optimizer; "Adam" runs
    # a 3DGS Adam baseline on the dense checkpoint's pipeline.
    OPTIONS: Dict[str, Tuple[str, bool]] = {
        "Learn2Splat (dense)": ("dense", False),
        "Learn2Splat (sparse)": ("sparse", False),
        "Adam (3DGS)": ("dense", True),
    }

    optimizer_dd = server.gui.add_dropdown("Optimizer", tuple(OPTIONS))

    # Optimization controls — applied to the picked OptGS at Start; frozen
    # while optimizing, unfrozen by Reset. opt_batch_size is capped at the
    # number of training views (the per-step view minibatch can't exceed them).
    n_train_views = int(train_bv["image"].shape[1])
    max_steps_input = server.gui.add_number(
        "Max steps", min=1, max=1000, step=1, initial_value=cfg.max_steps
    )
    batch_size_input = server.gui.add_number(
        "Opt batch size", min=1, max=n_train_views, step=1,
        initial_value=min(cfg.opt_batch_size, n_train_views),
    )
    strategy_dd = server.gui.add_dropdown(
        "Opt batch strategy", ("random", "sequential", "fps"),
        initial_value=cfg.opt_batch_strategy,
    )
    opt_controls = (max_steps_input, batch_size_input, strategy_dd)

    start_btn = server.gui.add_button("Start optimization")
    reset_btn = server.gui.add_button("Reset to initialization")
    status = server.gui.add_markdown("**initialized** — pick an optimizer, then Start")
    res_slider = (
        server.gui.add_slider(
            "Render height", min=240, max=1080, step=60, initial_value=540
        )
        if mode == "server"
        else None
    )

    init_gaussians = gaussians.clone()  # pristine copy, for Reset
    current = init_gaussians            # Gaussians currently displayed
    active = instances["dense"]         # OptGS used to render + to optimize next
    gen = None                          # optimize_iter generator while running
    last_cam_ts: dict = {}              # client id -> last-rendered camera stamp
    lock = threading.Lock()
    state = {
        "mode": "init",                 # "init" | "optimizing" | "done"
        "step": 0,
        "start": False,
        "reset": False,
        "rerender": False,              # a GUI control changed -> re-render once
        "selected": next(iter(OPTIONS)),
    }

    @start_btn.on_click
    def _(_) -> None:
        with lock:
            if state["mode"] in ("init", "done"):
                state["selected"] = optimizer_dd.value
                state["start"] = True

    @reset_btn.on_click
    def _(_) -> None:
        with lock:
            state["reset"] = True

    # The render-height slider only affects server-rendered frames; re-render
    # on change so the new resolution takes effect without a camera move.
    if res_slider is not None:

        @res_slider.on_update
        def _(_) -> None:
            with lock:
                state["rerender"] = True

    # Frame newly-connected clients on the first training camera (viser and
    # optgs share the OpenCV camera-to-world convention).
    cam_extr = train_bv["extrinsics"][0, 0].detach().cpu().numpy()

    @server.on_client_connect
    def _(client) -> None:
        try:
            client.camera.position = cam_extr[:3, 3]
            client.camera.wxyz = vtf.SO3.from_matrix(cam_extr[:3, :3]).wxyz
        except Exception:
            pass

    if mode == "client":  # show the initialization immediately
        # Black backdrop for the WebGL splat renderer (viser's canvas is not
        # black by default); on server.scene so late-joining clients get it.
        server.scene.set_background_image(np.zeros((8, 8, 3), dtype=np.uint8))
        server.scene.add_gaussian_splats(
            "/optgs/splats", **gaussians_to_splat_data(current)
        )

    console.print(
        f"[green]✓[/] viser GUI ([cyan]{mode}[/]) on port [cyan]{cfg.gui_port}[/]"
        f" — forward the port over SSH and open the printed URL"
    )

    try:
        while True:
            changed = False

            with lock:
                do_reset, do_start = state["reset"], state["start"]
                do_rerender = state["rerender"]
                state["reset"] = state["start"] = state["rerender"] = False
                selected = state["selected"]

            if do_rerender:
                changed = True  # server mode re-renders every connected client

            if do_reset:
                if gen is not None:
                    gen.close()  # runs optimize_iter's finally -> on_scene_end()
                    gen = None
                current = init_gaussians
                with lock:
                    state["mode"], state["step"] = "init", 0
                optimizer_dd.disabled = start_btn.disabled = False
                for c in opt_controls:
                    c.disabled = False
                changed = True

            if do_start and gen is None:
                name, use_adam = OPTIONS[selected]
                active = instances[name]
                # Apply the GUI optimization controls before the run starts.
                active.num_refine = int(max_steps_input.value)
                active.opt_batch_size = int(batch_size_input.value)
                active.opt_batch_strategy = strategy_dd.value
                opt = (
                    build_adam_baseline(active.num_refine).to(device)
                    if use_adam
                    else None
                )
                gen = active.optimize_iter(optimizer=opt)
                with lock:
                    state["mode"], state["step"] = "optimizing", 0
                optimizer_dd.disabled = start_btn.disabled = True
                for c in opt_controls:
                    c.disabled = True

            if gen is not None:
                try:
                    step, current = next(gen)
                    changed = True
                    with lock:
                        state["step"] = step + 1
                except StopIteration:
                    gen = None
                    with lock:
                        state["mode"] = "done"
                    optimizer_dd.disabled = start_btn.disabled = False

            if mode == "server":
                for cid, client in server.get_clients().items():
                    try:
                        cam_ts = client.camera.update_timestamp
                        if last_cam_ts.get(cid) != cam_ts or changed:
                            last_cam_ts[cid] = cam_ts
                            image = render_view(
                                active, current, client.camera,
                                res_slider.value, device, dtype,
                            )
                            client.scene.set_background_image(image, format="jpeg")
                    except Exception:
                        continue  # no camera message from this client yet
            elif changed:  # client mode — re-push splats when the Gaussians change
                server.scene.add_gaussian_splats(
                    "/optgs/splats", **gaussians_to_splat_data(current)
                )

            with lock:
                status.content = (
                    f"**{state['mode']}** — step "
                    f"{state['step']}/{active.num_refine} — "
                    f"{current.means.shape[1]} Gaussians"
                )

            if gen is None:
                time.sleep(1 / 30)  # idle: poll cameras at ~30 Hz
    except KeyboardInterrupt:
        if gen is not None:
            gen.close()
        console.print("\n[yellow]GUI stopped.[/]")


def run_gradio_gui(
    instances: dict,
    gaussians: Gaussians,
    train_bv: dict,
    cfg: Config,
    device: torch.device,
) -> None:
    """Interactive gradio GUI — a browser port of :func:`run_gui` (viser).

    gradio can't stream the camera back to Python, so there is no free-camera
    server rendering. Instead the optimization is *watched* as a streamed
    decoder render from a chosen training view (``gr.Image``, refreshed every
    step), and the finished scene is handed to an interactive ``gr.Model3D``
    splat viewer (orbit / zoom in the browser). The controls mirror the viser
    GUI: pick the optimizer (Learn2Splat dense/sparse or a 3DGS Adam baseline),
    set the step budget / view-minibatch size / sampling strategy, Start, Reset.

    ``instances`` maps "dense"/"sparse" to their initialized ``OptGS``.
    """
    import gc

    import gradio as gr

    from optgs.experimental.api.integration.config_bridge import build_adam_baseline
    from optgs.misc.image_io import prep_image
    from optgs.model.ply_export import save_gaussian_ply

    # Optimizer dropdown label -> (instances key, swap in a 3DGS Adam baseline);
    # mirrors run_gui's OPTIONS.
    OPTIONS: Dict[str, Tuple[str, bool]] = {
        "Learn2Splat (dense)": ("dense", False),
        "Learn2Splat (sparse)": ("sparse", False),
        "Adam (3DGS)": ("dense", True),
    }

    n_train_views = int(train_bv["image"].shape[1])
    h_full, w_full = train_bv["image"].shape[3], train_bv["image"].shape[4]
    init_gaussians = gaussians.clone()  # pristine copy; each Start re-inits from it

    # Shared state (single-GPU, single-session demo, like run_gui's globals): the
    # Gaussians currently shown, the OptGS rendering them, and a counter for
    # unique PLY filenames (so Model3D reloads instead of serving a stale cache).
    holder = {"current": init_gaussians, "active": instances["dense"], "ply": 0}

    ply_dir = os.path.join(cfg.result_dir, "gradio")
    os.makedirs(ply_dir, exist_ok=True)

    @torch.no_grad()
    def render_decoder(inst, g: Gaussians, view_idx: float, height: float) -> np.ndarray:
        """Decoder-render Gaussians ``g`` from training view ``view_idx``.

        Normalized intrinsics make the render resolution-independent; the width
        is derived from ``height`` at the training views' aspect ratio.
        """
        h = int(height)
        w = max(1, round(h * w_full / h_full))
        sl = slice(int(view_idx), int(view_idx) + 1)
        out = inst.decoder.forward(
            g,
            train_bv["extrinsics"][:, sl],
            train_bv["intrinsics"][:, sl],
            train_bv["near"][:, sl],
            train_bv["far"][:, sl],
            image_shape=(h, w),
        )
        return prep_image(out.color[0, 0])  # [H, W, 3] uint8

    def reorient_for_viewer(g: Gaussians) -> Gaussians:
        """Reorient a copy of ``g`` from the COLMAP world (this scene is Z-up)
        into the Y-up frame the gradio Model3D shows upright.

        gradio/Babylon flips the loaded splats' Y (``scaling.y *= -1``), so
        exporting through the reflection E(p)=(x,-z,-y) makes the *displayed*
        scene N(p)=(x,z,-y) — the world's Z-up mapped onto Babylon's Y-up. E's
        point-reflection part leaves the covariance unchanged; its proper-rotation
        part R_p=-E rotates the splat orientations to match.
        """
        from scipy.spatial.transform import Rotation as Rsp

        g = g.clone()
        m = g.means[0]
        g.means = torch.stack([m[:, 0], -m[:, 2], -m[:, 1]], dim=1)[None]  # E
        q = F.normalize(g.rotations_unnorm[0], dim=-1).detach().cpu().numpy()  # xyzw
        R_p = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        q = (Rsp.from_matrix(R_p) * Rsp.from_quat(q)).as_quat()  # R_p @ R, xyzw
        g.rotations_unnorm = torch.from_numpy(q).to(g.means)[None]
        return g

    def export_ply(g: Gaussians) -> str:
        """Write ``g`` (reoriented to Y-up) to a fresh PLY path for the viewer."""
        from pathlib import Path

        holder["ply"] += 1
        path = os.path.join(ply_dir, f"result_{holder['ply']}.ply")
        save_gaussian_ply(reorient_for_viewer(g), save_path=Path(path))
        return path

    def start(optimizer_label, max_steps, batch_size, strategy, view_idx, height):
        """Generator: run the picked optimizer, streaming a decoder render each
        step, then load the finished splats into the Model3D viewer."""
        name, use_adam = OPTIONS[optimizer_label]
        inst = instances[name]
        holder["active"] = inst
        # Re-init from the pristine copy so repeated Starts share one start point.
        inst.initialize_from_tensors(init_gaussians.clone(), train_bv)
        inst.num_refine = int(max_steps)
        inst.opt_batch_size = min(int(batch_size), n_train_views)
        inst.opt_batch_strategy = strategy
        opt = build_adam_baseline(inst.num_refine).to(device) if use_adam else None

        try:
            # Disable Start, hide the viewer, keep the placeholder while running.
            yield (
                render_decoder(inst, init_gaussians, view_idx, height),
                f"**optimizing** — step 0/{inst.num_refine} — "
                f"{init_gaussians.means.shape[1]} Gaussians",
                gr.update(visible=False, value=None),  # hide the viewer
                gr.update(interactive=False),          # disable Start
                gr.update(visible=True),               # keep the placeholder
            )

            g = init_gaussians
            for step, g in inst.optimize_iter(optimizer=opt):
                holder["current"] = g
                yield (
                    render_decoder(inst, g, view_idx, height),
                    f"**optimizing** — step {step + 1}/{inst.num_refine} — "
                    f"{g.means.shape[1]} Gaussians",
                    gr.update(), gr.update(), gr.update(),  # no change mid-run
                )

            yield (
                render_decoder(inst, g, view_idx, height),
                f"**done** — {inst.num_refine} steps — {g.means.shape[1]} Gaussians",
                gr.update(visible=True, value=export_ply(g)),  # reveal the viewer
                gr.update(interactive=True),                   # re-enable Start
                gr.update(visible=False),                      # hide the placeholder
            )
        finally:
            # Free the run's CUDA work (also runs if the user hits Stop mid-run),
            # so GPU memory doesn't accumulate across repeated runs.
            opt = None
            gc.collect()
            torch.cuda.empty_cache()

    def reset(view_idx, height):
        """Restore the initialization: re-render it, hide the viewer.

        No CUDA cleanup here — ``start``'s ``finally`` already frees each run's
        work; reset only re-renders the init.
        """
        holder["current"] = init_gaussians
        holder["active"] = instances["dense"]
        return (
            render_decoder(instances["dense"], init_gaussians, view_idx, height),
            "**Initialized.** Pick a method, then Start.",
            gr.update(visible=False, value=None),  # hide the viewer
            gr.update(interactive=True),           # enable Start
            gr.update(visible=True),               # show the placeholder
        )

    def rerender(view_idx, height):
        """Re-render the current Gaussians (preview view / height changed)."""
        return render_decoder(holder["active"], holder["current"], view_idx, height)

    initial_img = render_decoder(instances["dense"], init_gaussians, 0, 540)

    # Open the interactive viewer on the same vantage as the live render's
    # default preview (view 0). The viewer shows splats in the reoriented frame
    # N(p)=(x,z,-y) (see reorient_for_viewer); the orbit camera sits at the
    # training view's position and looks at the scene centroid. babylon_camera
    # maps the world camera into N and inverts Babylon's ArcRotateCamera position
    # formula rel=(r·cosα·sinβ, r·cosβ, r·sinα·sinβ) into (alpha°, beta°, radius).
    def babylon_camera(c2w: np.ndarray, centroid: np.ndarray) -> tuple:
        p = c2w[:3, 3] - centroid
        rel = np.array([p[0], p[2], -p[1]], dtype=np.float64)  # N applied to (cam - centroid)
        radius = float(np.linalg.norm(rel)) or 1e-3
        beta = float(np.degrees(np.arccos(np.clip(rel[1] / radius, -1.0, 1.0))))
        alpha = float(np.degrees(np.arctan2(rel[2], rel[0])))
        return (alpha, beta, radius)

    means0 = init_gaussians.means[0].detach().float().cpu().numpy()
    centroid0 = (means0.min(0) + means0.max(0)) / 2.0
    cam0_c2w = train_bv["extrinsics"][0, 0].detach().float().cpu().numpy()
    init_camera = babylon_camera(cam0_c2w, centroid0)

    # --- Visual style: lifted from the Learn2Splat project page
    # (https://naamapearl.github.io/learn2splat/) — plum accent (#B04080) with an
    # indigo hover, a light slate canvas with white cards, Source Serif 4
    # headings / Inter body / JetBrains Mono labels. ---
    plum = gr.themes.Color(
        c50="#faf0f6", c100="#f5e6ef", c200="#eccadd", c300="#dda3c2",
        c400="#c66ba0", c500="#b04080", c600="#9b3570", c700="#7f2a5b",
        c800="#6a2550", c900="#581f43", c950="#350e26", name="plum",
    )
    slate = gr.themes.Color(
        c50="#f7f8fc", c100="#eef0f8", c200="#e2e5ef", c300="#c8cde0",
        c400="#8890b0", c500="#5a6080", c600="#454b6b", c700="#343a56",
        c800="#262b42", c900="#1a1d2e", c950="#0f1120", name="slate",
    )
    theme = gr.themes.Soft(
        primary_hue=plum,
        neutral_hue=slate,
        font=["Inter", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f7f8fc",
        body_text_color="#1a1d2e",
        block_background_fill="#ffffff",
        block_border_color="#e2e5ef",
        block_radius="12px",
        block_label_text_color="#5a6080",
        block_title_text_color="#1a1d2e",
        border_color_primary="#e2e5ef",
        input_background_fill="#ffffff",
        button_primary_background_fill="#b04080",
        button_primary_background_fill_hover="#3d50c0",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_border_color="#c8cde0",
        button_secondary_text_color="#1a1d2e",
        slider_color="#b04080",
        link_text_color="#b04080",
    )
    # Source Serif 4 / Inter / JetBrains Mono, loaded into <head> like the page.
    fonts_head = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&"
        "family=Inter:wght@300;400;500;600&"
        'family=JetBrains+Mono:wght@400;500&display=swap">'
    )
    css = """
    .gradio-container { max-width: 1580px !important; margin: 0 auto !important; }
    #l2s-hero { background: linear-gradient(180deg,#f3f4fb 0%,#f7f8fc 100%);
        border:1px solid #e2e5ef; border-radius:14px; padding:26px 30px; margin-bottom:6px; }
    #l2s-hero .eyebrow { font-family:'JetBrains Mono',monospace; font-size:12px;
        letter-spacing:.16em; text-transform:uppercase; color:#b04080; font-weight:500;
        display:inline-flex; align-items:center; gap:8px; }
    #l2s-hero .eyebrow .dot { width:6px; height:6px; border-radius:50%; background:#b04080; }
    #l2s-hero h1 { font-family:'Source Serif 4',Georgia,serif; font-weight:600; color:#1a1d2e;
        font-size:clamp(1.55rem,3vw,2.25rem); line-height:1.2; margin:13px 0 10px; }
    #l2s-hero p { color:#5a6080; font-size:15px; line-height:1.6; margin:0; max-width:820px; }
    #l2s-hero a { color:#b04080; text-decoration:none; border-bottom:1px solid #e6cdd9;
        white-space:nowrap; }
    .l2s-eyebrow span { font-family:'JetBrains Mono',monospace; font-size:11px;
        letter-spacing:.15em; text-transform:uppercase; color:#8890b0; font-weight:500; }
    #l2s-start button, #l2s-reset button { font-family:'JetBrains Mono',monospace;
        letter-spacing:.03em; font-weight:500; }
    #l2s-status { background:#f7f8fc; border:1px solid #e2e5ef; border-left:3px solid #b04080;
        border-radius:9px; padding:4px 14px; }
    #l2s-status p { color:#5a6080; font-size:14px; margin:8px 0; }
    #l2s-status strong { color:#1a1d2e; }
    .l2s-ph { display:flex; align-items:center; justify-content:center; text-align:center;
        height:420px; border:1px dashed #c8cde0; border-radius:12px; background:#fbfcff;
        color:#8890b0; font-family:'JetBrains Mono',monospace; font-size:12.5px;
        letter-spacing:.04em; line-height:1.7; }
    footer { display:none !important; }
    """
    hero_html = (
        "<div class='eyebrow'><span class='dot'></span>Learn2Splat · Interactive demo</div>"
        "<h1>Extending the Horizon of Learned 3DGS Optimization</h1>"
        "<p>SfM-initialize a COLMAP scene, then refine the Gaussians with the "
        "meta-learned optimizer — pick a method, press <b>Start</b>, and watch the "
        "decoder render converge. The finished splats load in the interactive 3D "
        "viewer. <a href='https://naamapearl.github.io/learn2splat/' target='_blank' "
        "rel='noopener'>Project page&nbsp;↗</a></p>"
    )

    with gr.Blocks(
        title="Learn2Splat — Demo", theme=theme, css=css, head=fonts_head,
        analytics_enabled=False,
    ) as ui:
        gr.HTML(hero_html, elem_id="l2s-hero")
        with gr.Row(equal_height=False):
            # Column 1 — controls.
            with gr.Column(scale=3, min_width=300):
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Optimizer</span></div>")
                    optimizer_dd = gr.Dropdown(
                        list(OPTIONS), value=next(iter(OPTIONS)), label="Method"
                    )
                    with gr.Row():
                        max_steps_input = gr.Number(
                            value=cfg.max_steps, minimum=1, maximum=1000, step=1,
                            precision=0, label="Max steps",
                        )
                        batch_size_input = gr.Number(
                            value=min(cfg.opt_batch_size, n_train_views),
                            minimum=1, maximum=n_train_views, step=1, precision=0,
                            label="Opt batch size",
                        )
                    strategy_dd = gr.Dropdown(
                        ["random", "sequential", "fps"],
                        value=cfg.opt_batch_strategy, label="Batch strategy",
                    )
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Preview</span></div>")
                    view_slider = gr.Slider(
                        0, n_train_views - 1, value=0, step=1, label="Preview view"
                    )
                    height_slider = gr.Slider(
                        240, 1080, value=540, step=60, label="Render height"
                    )
                with gr.Row():
                    start_btn = gr.Button(
                        "Start optimization", variant="primary",
                        elem_id="l2s-start", scale=2,
                    )
                    reset_btn = gr.Button(
                        "Reset", variant="secondary", elem_id="l2s-reset", scale=1
                    )
                status_md = gr.Markdown(
                    "**Initialized.** Pick a method, then Start.", elem_id="l2s-status"
                )
            # Column 2 — live decoder render (streamed during optimization).
            with gr.Column(scale=5, min_width=380):
                image_out = gr.Image(
                    value=initial_img, label="Optimizer · live",
                    height=540, format="jpeg", interactive=False,
                )
            # Column 3 — interactive splats (hidden until a run finishes; a
            # placeholder holds the column so the 3-up layout stays balanced).
            with gr.Column(scale=5, min_width=380):
                model3d_out = gr.Model3D(
                    label="Refined splats · interactive", height=540,
                    visible=False, camera_position=init_camera,
                )
                placeholder = gr.HTML(
                    "<div class='l2s-ph'>The interactive 3D splats<br>"
                    "appear here once a run finishes.</div>"
                )

        start_inputs = [
            optimizer_dd, max_steps_input, batch_size_input, strategy_dd,
            view_slider, height_slider,
        ]
        gui_outputs = [image_out, status_md, model3d_out, start_btn, placeholder]
        # One shared GPU lane (concurrency_id) so Start / Reset / preview re-renders
        # never run on the GPU at the same time — overlapping runs were the path to
        # runaway VRAM growth.
        start_btn.click(
            start, inputs=start_inputs, outputs=gui_outputs, concurrency_id="gpu"
        )
        reset_btn.click(
            reset, inputs=[view_slider, height_slider], outputs=gui_outputs,
            concurrency_id="gpu",
        )
        # Slider release (not change) — re-render once when the user lets go.
        view_slider.release(
            rerender, [view_slider, height_slider], image_out, concurrency_id="gpu"
        )
        height_slider.release(
            rerender, [view_slider, height_slider], image_out, concurrency_id="gpu"
        )

    console.print(
        f"[green]✓[/] gradio GUI on port [cyan]{cfg.gui_port}[/]"
        f" — forward the port over SSH and open the printed URL"
    )
    ui.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=cfg.gui_port, share=False,
        show_error=True,
    )


def main(cfg: Config) -> None:
    # Fetch the demo scene on first run, before anything else touches it.
    ensure_data(cfg.data_dir)

    from optgs.experimental.api import OptGS, OptGSError
    from optgs.experimental.api.integration.config_bridge import build_adam_baseline

    os.makedirs(cfg.result_dir, exist_ok=True)
    device = torch.device(cfg.device)
    dtype = torch.float32

    console.rule("[bold cyan]OptGS demo[/]  ·  Learn2Splat vs Adam")

    # --- COLMAP scene, train/val split ---
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        verbose=False,
    )
    dataset = Dataset(parser)
    val_idx = [i for i in range(len(dataset)) if i % cfg.test_every == 0]
    train_idx = [i for i in range(len(dataset)) if i % cfg.test_every != 0]
    scene_scale = scene_extent(parser, cfg.global_scale)
    console.print(
        f"scene scale [cyan]{scene_scale:.4f}[/]  ·  "
        f"train [cyan]{len(train_idx)}[/]  ·  val [cyan]{len(val_idx)}[/]"
    )
    train_bv = build_batched_views(
        *collect_cameras(dataset, train_idx), scene_scale, device, dtype
    )

    # --- Interactive GUI: build both learned-optimizer checkpoints (dense and
    # sparse), initialize each, and hand off to the viser GUI instead of the
    # headless comparison. The GUI's Optimizer dropdown picks between them. ---
    if cfg.with_gui is not None:
        instances = {}
        for name in ("dense", "sparse"):
            try:
                instances[name] = OptGS(
                    checkpoint=CHECKPOINTS[name],
                    device=cfg.device,
                    num_refine=cfg.max_steps,
                    opt_batch_size=cfg.opt_batch_size,
                    opt_batch_strategy=cfg.opt_batch_strategy,
                    rasterize_mode=cfg.rasterize_mode,
                    eps2d=cfg.eps2d,
                )
            except OptGSError as e:
                console.print(f"[bold red]OptGS error ({name}):[/] {e}")
                raise SystemExit(1)

        # One SfM init shared by both checkpoints: dense and sparse get an
        # identical starting point, and the GUI shows a single initialization
        # regardless of which optimizer is picked.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        gaussians = sfm_initialization(
            parser, cfg, instances["dense"].sh_degree, device, dtype
        )
        for inst in instances.values():
            inst.initialize_from_tensors(gaussians, train_bv)

        if cfg.with_gui == "gradio":
            run_gradio_gui(instances, gaussians, train_bv, cfg, device)
        else:
            run_gui(instances, gaussians, train_bv, cfg, device, dtype)
        return

    val_c2w, val_Ks, val_images = collect_cameras(dataset, val_idx)
    val_bv = build_batched_views(val_c2w, val_Ks, val_images, scene_scale, device, dtype)

    results: dict = {}

    def finish(optgs, refined, name: str, elapsed: float) -> None:
        """Persist + evaluate one run's result under results/demo/<name>/."""
        out_dir = os.path.join(cfg.result_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        optgs.export_ply(os.path.join(out_dir, "point_cloud.ply"))
        ev = render_and_score(optgs, refined, val_bv, val_images, out_dir, device)
        results[name] = {
            "psnr": ev["psnr"], "time": elapsed,
            "num_views": ev["num_views"], "num_GS": int(refined.means.shape[1]),
        }
        console.print(
            f"[green]✓[/] [bold]{name}[/] — PSNR [cyan]{ev['psnr']:.3f}[/]  ·  "
            f"[cyan]{elapsed:.1f}s[/]  → [yellow]{out_dir}[/]"
        )

    # --- Learned optimizer (Learn2Splat): dense, then sparse ---
    optgs = None
    for name in ("dense", "sparse"):
        optgs = None  # free the previous instance before building the next
        torch.cuda.empty_cache()
        try:
            optgs = OptGS(
                checkpoint=CHECKPOINTS[name],
                device=cfg.device,
                num_refine=cfg.max_steps,
                opt_batch_size=cfg.opt_batch_size,
                opt_batch_strategy=cfg.opt_batch_strategy,
                rasterize_mode=cfg.rasterize_mode,
                eps2d=cfg.eps2d,
            )
        except OptGSError as e:
            console.print(f"[bold red]OptGS error ({name}):[/] {e}")
            raise SystemExit(1)
        # Seed *after* construction so dense and sparse get an identical SfM init.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        gaussians = sfm_initialization(parser, cfg, optgs.sh_degree, device, dtype)
        optgs.initialize_from_tensors(gaussians, train_bv)

        torch.cuda.synchronize()  # drain setup GPU work so it isn't timed
        tic = time.time()
        refined = optgs.optimize()
        torch.cuda.synchronize()
        finish(optgs, refined, name, time.time() - tic)

    # --- Fair Adam baseline: same SfM init / views / step budget / gsplat
    # renderer, run through the same optimize() path on the last OptGS
    # instance — only the update rule differs. ---
    adam = build_adam_baseline(optgs.num_refine).to(device)
    torch.cuda.synchronize()  # drain setup GPU work so it isn't timed
    tic = time.time()
    refined_adam = optgs.optimize(optimizer=adam)
    torch.cuda.synchronize()
    finish(optgs, refined_adam, "adam", time.time() - tic)

    # --- Comparison table ---
    table = Table(
        title=(
            f"Novel-view PSNR  ·  {results['dense']['num_views']} held-out "
            f"views  ·  {cfg.max_steps} steps  ·  "
            f"{results['dense']['num_GS']} Gaussians"
        ),
        title_style="bold",
        caption=(
            f"gsplat renderer  ·  "
            f"rasterize_mode={cfg.rasterize_mode or 'per-checkpoint'}  ·  "
            f"eps2d={cfg.eps2d if cfg.eps2d is not None else 'per-checkpoint'}"
        ),
    )
    table.add_column("Optimizer")
    table.add_column("PSNR (dB)", justify="right")
    table.add_column("Time (s)", justify="right")
    best = max(results, key=lambda k: results[k]["psnr"])
    for key, label in (
        ("dense", "Learn2Splat (dense)"),
        ("sparse", "Learn2Splat (sparse)"),
        ("adam", "Adam"),
    ):
        table.add_row(
            label,
            f"{results[key]['psnr']:.3f}",
            f"{results[key]['time']:.1f}",
            style="bold green" if key == best else None,
        )
    console.print(table)

    with open(os.path.join(cfg.result_dir, "stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓[/] results written to [yellow]{cfg.result_dir}[/]")


if __name__ == "__main__":
    main(tyro.cli(Config))
