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
    # Launch a viser GUI instead of the headless comparison. "server" renders
    # frames with the optgs decoder; "client" uses viser's built-in WebGL
    # Gaussian-splat renderer. Unset = headless run.
    with_gui: Optional[Literal["client", "server"]] = None
    # Port for the viser GUI web server (--with-gui only).
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
