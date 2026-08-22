"""Hydra-free checkpoint -> optimizer construction.

Rebuilds the learned optimizer (architecture + weights) from a checkpoint
*without* going through Hydra. Only ``_load_checkpoint_cfg`` +
``load_typed_config`` are used (both Hydra-free); the Hydra coupling lives in
``setup_cfg`` / ``merge_config_from_file`` / ``setup_output_dir`` which we never
call. All heavy imports are deferred into the functions so ``import optgs``
stays cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from optgs.experimental.api.integration.scene_protocol import OptGSError


@lru_cache(maxsize=8)
def _load_ckpt_cfg_cached(cfg_path_str: str):
    """Load + migrate a checkpoint config once per path (read-only callers).

    ``build_optimizer_cfg`` / ``build_decoder`` / ``get_scene_trainer_scalar``
    all need the same DictConfig; caching avoids re-parsing the file.
    """
    from optgs.config import _load_checkpoint_cfg  # Hydra-free

    return _load_checkpoint_cfg(Path(cfg_path_str))


def get_checkpoint_scalar(cfg_path: Path, key: str, default):
    """Read one scalar from the safely loaded checkpoint configuration."""
    from omegaconf import OmegaConf

    cfg = _load_ckpt_cfg_cached(str(cfg_path))
    return OmegaConf.select(cfg, key, default=default)


def get_scene_trainer_scalar(cfg_path: Path, key: str, default):
    """Read ``scene_trainer.<key>`` from a checkpoint config (or ``default``).

    Used for scalars that live on the (Hydra-free unavailable) scene-trainer
    config rather than the optimizer cfg: ``num_update_steps``,
    ``iter_batch_size``, ``sh_degree_interval``.
    """
    return get_checkpoint_scalar(cfg_path, f"scene_trainer.{key}", default)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

    from optgs.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizerCfg


def _optimizer_class_by_cfg_name():
    """Map a checkpoint config's ``scene_optimizer.name`` -> optimizer class.

    The registry (``SCENE_OPTIMIZERS``) keys on registry names (e.g.
    ``"depthsplat"``), but a checkpoint config's ``scene_optimizer.name`` is the
    cfg literal (``"knn_based"`` / ``"l2s"`` / ``"resplat_v1"`` /
    ``"resplat_v2"``). Each class asserts ``cfg.name`` is its ``OPTIMIZER_NAME``
    or one of its ``OPTIMIZER_NAME_ALIASES`` (e.g. legacy ``"clogs"`` for
    ``Learn2SplatOptimizer``), so we dispatch on both.
    """
    from optgs.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizer
    from optgs.scene_trainer.optimizer.optimizer_learn2splat import (
        Learn2SplatOptimizer,
    )
    from optgs.scene_trainer.optimizer.optimizer_resplat import (
        ResplatOptimizerV1,
        ResplatOptimizerV2,
    )

    classes = (
        KnnBasedOptimizer,
        Learn2SplatOptimizer,
        ResplatOptimizerV1,
        ResplatOptimizerV2,
    )
    mapping = {}
    for cls in classes:
        for name in (cls.OPTIMIZER_NAME, *getattr(cls, "OPTIMIZER_NAME_ALIASES", ())):
            mapping[name] = cls
    return mapping


def _initializer_cfg_class(name: str):
    """Map ``scene_initializer.name`` -> its concrete typed Cfg dataclass.

    ``InitializerCfg`` is a PEP-604 union; dacite needs a concrete dataclass
    as the top-level target (a union is only resolvable as a *field* type).
    Keyed to match both ``SCENE_INITIALIZERS`` and each Cfg's ``name``
    Literal.
    """
    from optgs.scene_trainer.initializer import (
        InitializerColmapCfg,
        InitializerEdgsCfg,
        InitializerPlyCfg,
        InitializerPointcloudCfg,
        InitializerRandomCfg,
        ResplatInitializerCfg,
    )

    return {
        "resplat_v1": ResplatInitializerCfg,
        "resplat_v2": ResplatInitializerCfg,
        "colmap": InitializerColmapCfg,
        "ply": InitializerPlyCfg,
        "edgs": InitializerEdgsCfg,
        "random": InitializerRandomCfg,
        "pointcloud": InitializerPointcloudCfg,
    }.get(name)


def _compose_default_group(group: str, value: str):
    """Hydra-compose the bundled default for ``scene_trainer.<group>=<value>``.

    Released checkpoints predate fields later added to the typed configs
    (e.g. ``scene_optimizer.refiner.fallback_means_lr``). The training/eval
    pipeline reconciles this by merging the checkpoint config over the
    *current* default config (config.py:merge_config_from_file). We mirror
    that: compose the bundled default for the group (e.g.
    ``scene_optimizer=knn_based`` -> base -> refiner:none, or
    ``scene_initializer=colmap``) so missing fields can be backfilled with
    current defaults while checkpoint values win for shared keys.

    Scoped use of ``hydra.compose`` (no ``@hydra.main`` / ``HydraConfig.get``,
    no app context); lazily imported so ``import optgs`` stays light. Returns
    ``None`` if composition fails (caller falls back to a strict parse).
    """
    try:
        import optgs
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        config_dir = str(Path(optgs.__file__).resolve().parent / "config")
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                composed = compose(
                    config_name="main",
                    overrides=[f"scene_trainer/{group}={value}"],
                )
        finally:
            GlobalHydra.instance().clear()
        return OmegaConf.select(composed, f"scene_trainer.{group}")
    except Exception as e:  # noqa: BLE001 - best-effort backfill
        print(
            f"[optgs] warning: could not compose default scene_trainer.{group}"
            f"={value} for back-compat merge ({type(e).__name__}: {e}); "
            f"parsing checkpoint config as-is."
        )
        return None


def build_optimizer_cfg(cfg_path: Path) -> tuple["KnnBasedOptimizerCfg", int | None]:
    """Load a checkpoint's saved config and return its typed optimizer cfg.

    Returns ``(KnnBasedOptimizerCfg, num_update_steps)`` where
    ``num_update_steps`` (the per-scene optimization step count) is read from
    ``scene_trainer.num_update_steps`` if present (it is NOT part of the
    optimizer cfg), else ``None``.
    """
    from omegaconf import OmegaConf

    from optgs.config import load_typed_config
    from optgs.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizerCfg

    cfg = _load_ckpt_cfg_cached(str(cfg_path))  # read_omega_cfg + migrate; NO Hydra
    so = OmegaConf.select(cfg, "scene_trainer.scene_optimizer")
    name = OmegaConf.select(cfg, "scene_trainer.scene_optimizer.name")
    if so is None or name in (None, "none"):
        raise OptGSError(
            f"checkpoint config at {cfg_path} has no learned scene_optimizer "
            f"(scene_trainer.scene_optimizer={name!r}). OptGS needs a learned "
            f"optimizer checkpoint (knn_based / clogs / resplat_v1 / resplat_v2)."
        )
    # Backfill fields a released (older) checkpoint config lacks with the
    # current defaults, then let checkpoint values win for shared keys
    # (mirrors config.py:merge_config_from_file's OmegaConf.merge).
    default_so = _compose_default_group("scene_optimizer", "knn_based")
    if default_so is not None:
        OmegaConf.set_struct(default_so, False)
        merged_so = OmegaConf.merge(default_so, so)
    else:
        merged_so = so
    try:
        opt_cfg = load_typed_config(merged_so, KnnBasedOptimizerCfg)
    except Exception as e:  # dacite/omegaconf errors -> actionable message
        raise OptGSError(
            f"failed to parse scene_optimizer from {cfg_path} into "
            f"KnnBasedOptimizerCfg ({type(e).__name__}: {e})."
        ) from e

    # Mirror SceneTrainerCfg (scene_trainer_cfg.py: scene_optimizer.update(
    # scene_initializer)): wire the checkpoint's initializer cfg into the
    # optimizer cfg so the runtime-only fields init_sh_d / sh_d — absent from
    # every config file — are populated before the optimizer nn.Module is built.
    si = OmegaConf.select(cfg, "scene_trainer.scene_initializer")
    si_name = OmegaConf.select(cfg, "scene_trainer.scene_initializer.name")
    if si is None or si_name in (None, "none"):
        raise OptGSError(
            f"checkpoint config at {cfg_path} has no scene_initializer "
            f"(name={si_name!r}); cannot derive the optimizer's initializer "
            f"settings required to build it."
        )
    init_cls = _initializer_cfg_class(str(si_name))
    if init_cls is None:
        raise OptGSError(
            f"unsupported scene_initializer.name={si_name!r} in {cfg_path}; "
            f"cannot derive the optimizer's initializer settings."
        )
    default_si = _compose_default_group("scene_initializer", str(si_name))
    if default_si is not None:
        OmegaConf.set_struct(default_si, False)
        merged_si = OmegaConf.merge(default_si, si)
    else:
        merged_si = si
    try:
        init_cfg = load_typed_config(merged_si, init_cls)
        opt_cfg.update(init_cfg)  # sets init_sh_d/sh_d
    except Exception as e:
        raise OptGSError(
            f"failed to wire scene_initializer ({si_name!r}) into the "
            f"optimizer cfg from {cfg_path} ({type(e).__name__}: {e})."
        ) from e

    num_update_steps = OmegaConf.select(
        cfg, "scene_trainer.num_update_steps", default=None
    )
    return opt_cfg, num_update_steps


def build_decoder(
    cfg_path: Path, dataset_cfg: object, decoder_overrides: dict | None = None
) -> "nn.Module":
    """Build the renderer the checkpoint was trained with.

    Uses ``scene_trainer.decoder`` from the checkpoint config (NOT a hardcoded
    backend): the learned optimizer's in-loop render gradients must match the
    backend it trained with, and only the registered/available backends are
    usable (e.g. ``gsplat`` — the optgs default; the ``inria`` backend needs
    ``diff_gaussian_rasterization``, which is optional). ``dataset_cfg`` only
    needs a ``background_color`` attribute. ``decoder_overrides`` (e.g.
    ``rasterize_mode`` / ``eps2d``) take precedence over the checkpoint config.
    """
    from omegaconf import OmegaConf

    from optgs.config import load_typed_config
    from optgs.model.decoder import DECODER_CFGS, get_decoder

    cfg = _load_ckpt_cfg_cached(str(cfg_path))
    node = OmegaConf.select(cfg, "scene_trainer.decoder")
    if node is None:
        raise OptGSError(
            f"checkpoint config at {cfg_path} has no scene_trainer.decoder; "
            f"cannot rebuild the renderer the optimizer trained with."
        )
    overrides = dict(decoder_overrides or {})
    requested_backend = overrides.pop("name", None)
    if requested_backend is not None:
        default_node = _compose_default_group("decoder", str(requested_backend))
        if default_node is None:
            raise OptGSError(
                f"could not compose decoder backend override {requested_backend!r}"
            )
        node = default_node

    # Resolve the discriminated union by `name` (dacite can't parse the union
    # itself). A missing name means its optional backend isn't installed.
    name = OmegaConf.select(node, "name")
    cfg_cls = DECODER_CFGS.get(name)
    if cfg_cls is None:
        raise OptGSError(
            f"decoder backend {name!r} (from {cfg_path}) is not available in "
            f"this environment. Install its backend (e.g. "
            f"diff_gaussian_rasterization for 'inria') or use a checkpoint "
            f"trained with the 'gsplat' decoder."
        )
    # gsplat decoder rasterize_mode / eps2d, by precedence:
    #   caller override  >  checkpoint config  >  gsplat rasterization() default
    # (so an older checkpoint that omits a field behaves as plain gsplat would).
    if name == "gsplat":
        import inspect

        from gsplat.rendering import rasterization

        sig = inspect.signature(rasterization).parameters
        node = OmegaConf.merge(
            OmegaConf.create(
                {f: sig[f].default for f in ("rasterize_mode", "eps2d") if f in sig}
            ),
            node,
            OmegaConf.create(overrides),
        )
    elif overrides:
        node = OmegaConf.merge(node, OmegaConf.create(overrides))
    try:
        decoder_cfg = load_typed_config(node, cfg_cls)
    except Exception as e:
        raise OptGSError(
            f"failed to parse scene_trainer.decoder from {cfg_path} "
            f"({type(e).__name__}: {e})."
        ) from e
    return get_decoder(decoder_cfg, dataset_cfg)


def build_optimizer(opt_cfg: "KnnBasedOptimizerCfg") -> "nn.Module":
    """Construct the concrete learned optimizer for ``opt_cfg`` (no weights)."""
    from optgs.misc.io import FrequencyScheduler

    mapping = _optimizer_class_by_cfg_name()
    cls = mapping.get(opt_cfg.name)
    if cls is None:
        raise OptGSError(
            f"unsupported scene_optimizer.name={opt_cfg.name!r}; OptGS supports "
            f"{sorted(mapping)}."
        )
    optimizer = cls(opt_cfg)
    # The optimizer's save_every (info/context/target/debug artifact dumps) is
    # wired by SceneTrainer during training; the optimizer calls it
    # unconditionally, so the API inference path — which has nothing to dump —
    # installs a disabled scheduler instead of leaving it None.
    save_every = FrequencyScheduler(last_step=0)
    save_every.disable(True)
    optimizer.save_every = save_every
    return optimizer


def build_refiner_cfg(strategy: str, num_refine: int) -> "BaseStrategyCfg":
    """Load the bundled ``refiner/{strategy}.yaml`` and, when it densifies, rescale
    its schedule so ADC actually fires within ``num_refine`` steps.

    ``strategy`` is a raw refiner name ("none"/"default"/"edgs"/"mcmc"/… — a sibling
    .yaml under ``config/scene_trainer/scene_optimizer/refiner/``). The bundled
    configs use the stock 3DGS schedule (densify from iter 500), which never
    triggers in the demo's ~100-step runs, so for any densifying strategy we set
    start ≈ 10%·N, interval ≈ 20%·N, stop = N, and bound ``cap_max`` for VRAM safety.
    ``reset_every`` is left at its (large) bundled value so opacity reset does not
    thrash a short run. Returns a ``BaseStrategyCfg``.
    """
    from dataclasses import replace

    import optgs
    from omegaconf import OmegaConf

    from optgs.config import load_typed_config
    from optgs.scene_trainer.adc.fastgs import FastGSStrategyCfg
    from optgs.scene_trainer.adc.legs_config import LeGSStrategyCfg
    from optgs.scene_trainer.adc.mcmc import McmcStrategyCfg
    from optgs.scene_trainer.adc.vanilla import AdaptiveStrategyCfg, VanillaStrategyCfg

    cfg_dir = Path(optgs.__file__).resolve().parent / "config"
    yaml_path = (
        cfg_dir / "scene_trainer" / "scene_optimizer" / "refiner" / f"{strategy}.yaml"
    )
    if not yaml_path.exists():
        avail = sorted(p.stem for p in yaml_path.parent.glob("*.yaml"))
        raise OptGSError(
            f"unknown ADC strategy {strategy!r}: no refiner config at {yaml_path} "
            f"(available: {', '.join(avail)})."
        )
    # Build the matching StrategyCfg arm (by name) so the strategy-specific fields are present;
    # a bare BaseStrategyCfg would lack noise_lr (mcmc) / grad_abs_thresh (fastgs).
    arm = {
        "mcmc": McmcStrategyCfg,
        "fastgs": FastGSStrategyCfg,
        "adaptive": AdaptiveStrategyCfg,
        "adaptive_legacy": AdaptiveStrategyCfg,
        "legs": LeGSStrategyCfg,
        "legs_blur": LeGSStrategyCfg,
    }.get(strategy, VanillaStrategyCfg)
    cfg = load_typed_config(OmegaConf.load(yaml_path), arm)

    # Non-densifying strategies (e.g. "none") leave the Gaussian set fixed — no rescale.
    if not (cfg.do_densify or cfg.do_prune or cfg.do_opacity_reset):
        return cfg

    n = max(1, int(num_refine))
    cap = cfg.cap_max if (cfg.cap_max and cfg.cap_max > 0) else 1_500_000
    if cfg.name in {"legs", "legs_blur"}:
        # Exact LeGS transplantation keeps the released 500/100/15000
        # densification schedule and the absence of a global primitive cap.
        # Scaling these values to a smoke-test horizon is the older adapted
        # ablation, not the reference mechanism requested by --adc legs.
        return cfg
    if cfg.name == "adaptive":
        return replace(
            cfg,
            refine_start_iter=max(1, round(0.05 * n)),
            refine_every=max(1, round(0.10 * n)),
            refine_stop_iter=max(1, round(0.80 * n)),
            cap_max=cap,
        )
    return replace(
        cfg,
        refine_start_iter=max(1, round(0.1 * n)),
        refine_every=max(1, round(0.2 * n)),
        refine_stop_iter=n,
        cap_max=cap,
    )


def build_adam_baseline(
    num_refine: int,
    adc: str = "none",
    *,
    reward_conditioned: bool = False,
) -> "nn.Module":
    """Build the codebase's 3DGS Adam optimizer for a fair baseline comparison.

    Uses the bundled ``scene_optimizer=3dgs`` config — gsplat's example
    hyperparameters (LRs, betas) — with the means-LR decay horizon set to
    ``num_refine``. ``adc`` selects the densification strategy: the default
    ``"none"`` disables ADC so the baseline refines the same fixed Gaussian set as
    the learned optimizer (a like-for-like update-rule comparison); any other raw
    refiner name swaps in that strategy's schedule (scaled to ``num_refine`` via
    ``build_refiner_cfg``). Returns a ready-to-run ``AdamOptimizer``.
    """
    from dataclasses import asdict

    from omegaconf import OmegaConf

    from optgs.config import load_typed_config
    from optgs.misc.io import FrequencyScheduler
    from optgs.scene_trainer.optimizer.optimizer_adam import (
        AdamOptimizer,
        AdamOptimizerCfg,
    )

    composed = _compose_default_group("scene_optimizer", "3dgs")
    if composed is None:
        raise OptGSError(
            "could not Hydra-compose the bundled 'scene_optimizer=3dgs' config "
            "for the Adam baseline."
        )
    OmegaConf.set_struct(composed, False)
    # gsplat decays the means LR over the full step budget.
    composed.lr_scheduler.max_steps = int(num_refine)
    if adc == "none":
        # Disable densification — the baseline refines the same fixed Gaussian set
        # as the learned optimizer (a like-for-like comparison of the update rule).
        for flag in ("do_densify", "do_prune", "do_opacity_reset"):
            if flag in composed.refiner:
                composed.refiner[flag] = False
    else:
        # Swap in the chosen ADC strategy (schedule scaled to num_refine).
        refiner = build_refiner_cfg(adc, num_refine)
        if reward_conditioned:
            from dataclasses import replace

            from optgs.scene_trainer.adc.vanilla import AdaptiveStrategyCfg

            if not isinstance(refiner, AdaptiveStrategyCfg):
                raise OptGSError(
                    "reward-conditioned densification requires adc='adaptive'"
                )
            refiner = replace(refiner, reward_conditioned=True)
        composed.refiner = asdict(refiner)
    try:
        adam_cfg = load_typed_config(composed, AdamOptimizerCfg)
    except Exception as e:
        raise OptGSError(
            f"failed to parse the bundled '3dgs' config into AdamOptimizerCfg "
            f"({type(e).__name__}: {e})."
        ) from e

    optimizer = AdamOptimizer(adam_cfg)
    save_every = FrequencyScheduler(last_step=0)  # nothing to dump (see build_optimizer)
    save_every.disable(True)
    optimizer.save_every = save_every
    # AdamOptimizer is a NonlearnedOptimizer — already pinned to eval mode.
    return optimizer


def load_optimizer_state(
    optimizer: "nn.Module",
    ckpt_path: str,
    init_state_wo_features: bool,
    strict: bool,
) -> None:
    """Load optimizer weights from ``ckpt_path`` into ``optimizer``.

    Transcribes the prefix-stripping / legacy-rename / feature-drop logic from
    ``optgs/main.py:load_optimizer`` (we cannot call that function: it needs a
    full Hydra ``cfg`` and a ``scene_trainer``).
    """
    import torch

    from optgs.misc.checkpointing import _rename_optimizer_attrs

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip the Lightning "scene_trainer." prefix if present.
    state = {k.replace("scene_trainer.", ""): v for k, v in state.items()}

    if any(k.startswith("optimizer.") for k in state):
        # Unified repo format: keys are optimizer.*
        osd = {
            k[len("optimizer."):]: v
            for k, v in state.items()
            if k.startswith("optimizer.")
        }
    else:
        # Legacy Resplat format: keys are encoder.* (before init/opt split).
        osd = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }

    # Rename module attributes renamed in the update_/refine_ cleanup pass
    # (and the resplat-era render_error_mv_attn).
    osd = _rename_optimizer_attrs(osd)

    if not osd:
        raise OptGSError(
            f"no optimizer weights found in {ckpt_path} (looked for "
            f"'optimizer.*' or legacy 'encoder.*' keys)."
        )

    if init_state_wo_features:
        osd = {k: v for k, v in osd.items() if "state_proj" not in k}

    optimizer.load_state_dict(osd, strict=strict)
