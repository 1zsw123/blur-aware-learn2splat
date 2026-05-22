import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, Any, Callable

import hydra
import torch
from dacite import Config, from_dict, UnionMatchError
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy

from .config_migrate import migrate, CURRENT_CFG_VERSION
from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .global_cfg import set_cfg
from .loss import LossCfgWrapper
from .misc.io import CustomPath
from .misc.io import cyan, read_omega_cfg
from .misc.checkpointing import find_latest_ckpt
from .misc.hf_ckpt import maybe_resolve_hf_ref
from .paths import CKPT_DIR, RESULTS_DIR
from .scene_trainer.scene_trainer_cfg import SceneTrainerCfg, MetaOptimizerCfg, TestCfg, TrainCfg


# In order to extract filename or dirname from a path in the config
def checkpoint_rel_dir(path):
    rel_dir = CustomPath(path) - CKPT_DIR  # dir_path / checkpoints / epoch_x-step_xxxxx.ckpt
    dir_path = rel_dir.parent.parent
    return str(dir_path)


OmegaConf.register_new_resolver("checkpoint_rel_dir", checkpoint_rel_dir)
OmegaConf.register_new_resolver("parent_dir", lambda path: str(CustomPath(path).parent))


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    pretrained_monodepth: Optional[str]
    pretrained_mvdepth: Optional[str]
    pretrained_depth: Optional[str]
    pretrained_scale_predictor: Optional[str]
    pretrained_depth_teacher: Optional[str]
    no_strict_load: bool
    resume: bool
    no_resume_upsampler: bool
    partial_load: bool
    freeze_mono_vit: bool
    pretrained_initializer: Optional[str]
    pretrained_optimizer: Optional[str]
    resume_update_module: str | None
    load_existing_cfg: bool

    def __post_init__(self):
        # Resolve any Hugging Face Hub references (hf://org/repo/file[@rev]) to
        # local cached paths so all downstream torch.load calls work unchanged.
        for attr in ("pretrained_model", "pretrained_optimizer", "pretrained_initializer",
                     "pretrained_monodepth", "pretrained_mvdepth", "pretrained_depth",
                     "pretrained_scale_predictor", "pretrained_depth_teacher",
                     "resume_update_module"):
            resolved = maybe_resolve_hf_ref(getattr(self, attr))
            if resolved != getattr(self, attr):
                setattr(self, attr, resolved)

        for attr in ("pretrained_model", "pretrained_optimizer", "pretrained_initializer"):
            path = getattr(self, attr)
            if path is not None and Path(path).name == "last":
                try:
                    resolved = find_latest_ckpt(Path(path).parent)
                    setattr(self, attr, resolved)
                    print(f"Replacing {attr} to last checkpoint: {resolved}")
                except Exception as e:
                    print(cyan(f"Warning: {e}. Continuing with 'last' as {attr}."))


@dataclass
class MetaTrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: int
    eval_index: str | None
    limit_test_batches: int | float
    limit_train_batches: int | float
    test: TestCfg
    train: TrainCfg

    def get_dist_strategy(self, scene_trainer_cfg: SceneTrainerCfg):
        from .scene_trainer.initializer.initializer_resplat import ResplatInitializerCfg
        dist_strategy = "auto"
        if torch.cuda.device_count() > 1:
            dist_strategy = 'ddp'
            if isinstance(scene_trainer_cfg.scene_optimizer, ResplatInitializerCfg):
                if scene_trainer_cfg.scene_initializer.use_gt_depth:
                    dist_strategy = 'ddp_find_unused_parameters_true'
                if scene_trainer_cfg.scene_initializer.use_checkpointing or scene_trainer_cfg.scene_initializer.init_use_checkpointing:
                    dist_strategy = DDPStrategy(static_graph=True)
                if scene_trainer_cfg.use_fsdp:
                    def only_wrap_trainable(module, recurse, nonwrapped_numel):
                        has_trainable = any(p.requires_grad for p in module.parameters())
                        return has_trainable

                    dist_strategy = FSDPStrategy(auto_wrap_policy=only_wrap_trainable)
        if self.train.use_replay_buffer:
            # When resampling from the replay buffer,
            # we don't project the condition_features to state, so the update_proj is not used
            dist_strategy = "ddp_find_unused_parameters_true"
        return dist_strategy


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    scene_trainer: SceneTrainerCfg
    meta_optimizer: MetaOptimizerCfg  ## TODO Naama: should we move under meta trainer config?
    checkpointing: CheckpointingCfg
    meta_trainer: MetaTrainerCfg
    loss: list[LossCfgWrapper]
    seed: int
    use_plugins: bool
    output_dir: str
    version: int | None
    debug_cfg: bool

    def __post_init__(self):
        if self.mode == "test":
            self._setup_test_output_dir()

    def _setup_test_output_dir(self):
        base_res_dir = RESULTS_DIR
        if self.meta_trainer.limit_test_batches != 1.0:
            base_res_dir = RESULTS_DIR + f"_{self.meta_trainer.limit_test_batches}_scenes"
        if self.output_dir == "placeholder":
            if self.meta_trainer.test.postprocessing is not None and self.meta_trainer.test.postprocessing.is_active:
                self.output_dir = (base_res_dir /
                                   "nonlearned" /
                                   "vanilla_3dgs" /
                                   self.meta_trainer.test.postprocessing.name /
                                   self.meta_trainer.test.postprocessing.get_dir_name(with_name=False))
            else:
                ckpt_path = self.checkpointing.pretrained_model or self.checkpointing.pretrained_optimizer
                pretrained_model_rel_dir = checkpoint_rel_dir(ckpt_path)
                self.output_dir = (base_res_dir /
                                   "optgs" /
                                   pretrained_model_rel_dir)
        elif 'experimental' in str(self.output_dir):  # TODO (release): remove
            self._setup_experimental_output_dir()

    def _setup_experimental_output_dir(self):
        resplat_str = []
        grad_str = []
        normgrad_str = []
        assert self.scene_trainer.scene_optimizer.experimental_run
        for p in self.scene_trainer.scene_optimizer.experimental_update.param_names:
            update = getattr(self.scene_trainer.scene_optimizer.experimental_update, p)
            use_norm_grad = getattr(self.scene_trainer.scene_optimizer.experimental_use_norm_grads, p)
            use_grad = self.scene_trainer.scene_optimizer.experimental_use_grads and not use_norm_grad
            use_resplat = update and not use_grad and not use_norm_grad
            if update:
                assert use_grad ^ use_norm_grad ^ use_resplat, f"Invalid combination for {p}: use_resplat={use_resplat}, use_grad={use_grad}, use_norm_grad={use_norm_grad}"
                if use_resplat:
                    resplat_str.append(p)
                if use_grad:
                    grad_str.append(p)
                if use_norm_grad:
                    normgrad_str.append(p)

        if len(resplat_str) == len(self.scene_trainer.scene_optimizer.experimental_update.param_names):
            resplat_str = ["all"]
        if len(grad_str) == len(self.scene_trainer.scene_optimizer.experimental_update.param_names):
            grad_str = ["all"]
        if len(normgrad_str) == len(self.scene_trainer.scene_optimizer.experimental_update.param_names):
            normgrad_str = ["all"]

        exp_name = "_".join([
            ("resplat_" + "_".join(resplat_str) if len(resplat_str) > 0 else ""),
            ("grad_" + "_".join(grad_str) if len(grad_str) > 0 else ""),
            ("normgrad_" + "_".join(normgrad_str) if len(normgrad_str) > 0 else ""),
        ])

        output_dir_str = str(self.output_dir)
        output_dir_str = output_dir_str.replace("experimental", f"experimental_{exp_name}")
        self.output_dir = Path(output_dir_str)
        print(cyan(f"Experimental run, setting output_dir to {CustomPath(self.output_dir)}"))


TYPE_HOOKS = {
    Path: Path,
}

T = TypeVar("T")


def get_class_by_path(path: str):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _diagnose_union_error(e: UnionMatchError, data: dict, dacite_config: Config) -> str:
    """Try each union member individually and report per-member errors."""
    import dataclasses
    import typing
    union_type = e.field_type
    # Extract the member types from the union
    args = typing.get_args(union_type)
    if not args:
        return str(e)
    lines = [str(e), "", "Per-member diagnostics:"]
    for member_type in args:
        try:
            from_dict(member_type, data, config=dacite_config)
            lines.append(f"  {member_type.__name__}: matched OK (unexpected)")
        except Exception as member_err:
            lines.append(f"  {member_type.__name__}: {member_err}")
            # For dataclasses, also check for extra/missing fields
            if dataclasses.is_dataclass(member_type):
                expected = {f.name for f in dataclasses.fields(member_type)}
                provided = set(data.keys()) if isinstance(data, dict) else set()
                missing = expected - provided
                extra = provided - expected
                if missing:
                    lines.append(f"    missing fields: {missing}")
                if extra:
                    lines.append(f"    extra fields (ignored with strict=False): {extra}")
    return "\n".join(lines)


def load_typed_config(
        cfg: DictConfig,
        data_class: Type[T],
        extra_type_hooks: dict = {},
) -> T:
    dacite_config = Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks})
    try:
        return from_dict(
            data_class,
            OmegaConf.to_container(cfg),
            config=dacite_config,
        )
    except UnionMatchError as e:
        diagnostic = _diagnose_union_error(e, e.value, dacite_config)
        print(f"\n{'='*60}\n"
              f"Current config: {e.value}\n"
              "\n"
              "\n"
              f"UnionMatchError diagnostic:\n{diagnostic}\n{'='*60}"
              f"\n",
              flush=True)
        raise


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def universal_target_hook(cfg: dict, _: Type) -> Any:
    """Generic hook to construct config objects from `__target__`."""
    if not isinstance(cfg, dict):
        return None
    if "__target__" not in cfg:
        return None  # Let decite handle it

    cfg_copy = deepcopy(cfg)  # avoid mutating original
    target = cfg_copy.pop("__target__")

    if isinstance(target, str):
        target_type = get_class_by_path(target)
    else:
        target_type = target

    # Use recursive loading with known additional hooks
    return load_typed_config(
        DictConfig(cfg_copy),
        target_type,
    )


def make_target_hook_for_type(t: Type) -> Callable:
    return lambda cfg: universal_target_hook(cfg, t)


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    # scene_trainer/scene_optimizer=none loads a full dict from none.yaml;
    # dacite can't match that dict to the None arm of SceneOptimizerCfg | None.
    # Convert it to Python None here so dacite matches correctly.
    scene_opt = OmegaConf.select(cfg, "scene_trainer.scene_optimizer")
    if isinstance(scene_opt, DictConfig) and OmegaConf.select(scene_opt, "name") == "none":
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, "scene_trainer.scene_optimizer", None, merge=False)
        OmegaConf.set_struct(cfg, True)

    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers}
    )


def should_run(cfg_dict):
    if cfg_dict.mode == "test":
        if cfg_dict.meta_trainer.test.skip_if_outputs_exist:
            output_dir = cfg_dict.output_dir
            if not output_dir.exists():
                return True
            metrics_path_pattern = output_dir / "metrics" / "target_*_psnr.json"
            metric_paths = list(metrics_path_pattern.parent.glob(metrics_path_pattern.name))
            if len(metric_paths) > 0:
                print(cyan(f"Test metrics already exist at {metric_paths}."))
                return False
    return True


def setup_cfg(cfg_dict):
    # Get the original config from the output directory, when testing or resuming.
    cfg_dict = merge_config_from_file(cfg_dict)
    eval_cfg = get_eval_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    # Set global cfg object.
    set_cfg(cfg_dict)
    # Set up the output directory.
    setup_output_dir(cfg, cfg_dict)
    return cfg, cfg_dict, eval_cfg  # TODO Naama: why do we need both cfg and cfg_dict?


def flatten_wandb(cfg):
    """Recursively replace {'desc': ..., 'value': v} with v."""
    if isinstance(cfg, dict):
        if "value" in cfg and len(cfg) == 2 and "desc" in cfg:
            return flatten_wandb(cfg["value"])
        return {k: flatten_wandb(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [flatten_wandb(v) for v in cfg]
    else:
        return cfg


def _apply_cli_overrides(merged_cfg: DictConfig, orig_cli_cfg: DictConfig, raw_overrides: list[str]) -> DictConfig:
    """
    Re-apply CLI overrides onto merged_cfg after the checkpoint config has been merged in.

    Takes already-composed values from orig_cli_cfg rather than re-parsing the raw override
    strings. This correctly handles:
    - Group overrides (e.g. dataset/view_sampler=evaluation) → replace subtree from cli
    - Complex values (e.g. loss=[mse,ssim]) → replace subtree from cli
    - Interpolated values (e.g. output_dir=${...}) → take resolved value from cli
    - Defaults-list overrides (+experiment=re10k) → skip (already baked into orig_cli_cfg)
    """
    if not raw_overrides:
        return merged_cfg

    from hydra.core.override_parser.overrides_parser import OverridesParser
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(raw_overrides)

    print(cyan(f"Re-applying {len(raw_overrides)} CLI overrides onto merged config."))
    OmegaConf.set_struct(merged_cfg, False)

    # Architecture subtrees: CLI group default fills in *new* fields only;
    # checkpoint values win for fields that already exist.
    ARCH_KEYS = {"scene_optimizer", "scene_initializer"}
    # Sub-keys within ARCH_KEYS where CLI should always win over checkpoint values.
    CLI_WINS_SUBKEYS = {"refiner"}

    for override in parsed:
        key = override.key_or_group
        dotkey = key.replace("/", ".")

        cli_val = OmegaConf.select(orig_cli_cfg, dotkey, default=None, throw_on_resolution_failure=False)

        if cli_val is None:
            # No direct config path — e.g. +experiment=re10k is a defaults-list override
            # whose effect is already baked into orig_cli_cfg; nothing to apply.
            print(cyan(f"  Skipping '{key}' (no direct config path in cli)"))
            continue

        # For architecture group overrides: fill in missing fields from CLI defaults
        # without overriding checkpoint values for fields that already exist.
        is_group_override = "/" in key or isinstance(cli_val, (DictConfig, dict, list))
        if is_group_override and any(arch_key in dotkey for arch_key in ARCH_KEYS):
            # If the override targets a CLI-wins sub-key directly, CLI wins entirely.
            dotkey_parts = set(dotkey.split("."))
            if dotkey_parts & CLI_WINS_SUBKEYS:
                OmegaConf.update(merged_cfg, dotkey, cli_val, merge=False)
                print(cyan(f"  '{dotkey}': replace from cli (CLI wins)"))
                continue

            existing_val = OmegaConf.select(merged_cfg, dotkey, default=None)
            if existing_val is not None:
                # cli_val provides new defaults; existing_val (checkpoint) wins for shared fields
                new_val = OmegaConf.merge(cli_val, existing_val)
                # Re-apply CLI-wins sub-keys so they override checkpoint values.
                for subkey in CLI_WINS_SUBKEYS:
                    cli_subval = OmegaConf.select(cli_val, subkey, default=None)
                    if cli_subval is not None:
                        OmegaConf.set_struct(new_val, False)
                        OmegaConf.update(new_val, subkey, cli_subval, merge=False)
                        print(cyan(f"  '{dotkey}.{subkey}': CLI override applied (CLI wins)"))
                OmegaConf.update(merged_cfg, dotkey, new_val, merge=False)
                print(cyan(f"  '{dotkey}': fill-missing from cli (checkpoint values preserved)"))
                continue

        # Group overrides and complex values replace the whole subtree;
        # scalars are merged so sibling keys are preserved.
        replace = is_group_override
        print(cyan(f"  '{dotkey}': {'replace' if replace else 'update'} from cli"))
        OmegaConf.update(merged_cfg, dotkey, cli_val, merge=not replace)

    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


def _print_cfg_diff(before: dict, after: dict, prefix: str = "") -> None:
    """Recursively print keys that differ between two plain-dict config snapshots."""
    all_keys = set(before) | set(after)
    diffs = []
    for k in sorted(all_keys):
        full_key = f"{prefix}.{k}" if prefix else k
        b_val = before.get(k, "<missing>")
        a_val = after.get(k, "<missing>")
        if isinstance(b_val, dict) and isinstance(a_val, dict):
            _print_cfg_diff(b_val, a_val, prefix=full_key)
        elif b_val != a_val:
            diffs.append((full_key, b_val, a_val))
    for full_key, b_val, a_val in diffs:
        print(cyan(f"  [cfg diff]  {full_key}:  {b_val!r}  →  {a_val!r}"))


def _find_config_for_checkpoint(ckpt_path) -> Path | None:
    """Return the config.yaml path for a given checkpoint, or None."""
    p = Path(ckpt_path).parent.parent / "config.yaml"
    if p.exists():
        return p
    # Fall back to wandb latest-run
    p = Path(ckpt_path).parent.parent / "wandb" / "latest-run" / "files" / "config.yaml"
    if p.exists():
        return p
    return None


def _load_checkpoint_cfg(config_path: Path) -> DictConfig:
    """Load, migrate, and (if from wandb) flatten a checkpoint config file."""
    cfg = read_omega_cfg(config_path)
    cfg = migrate(cfg)
    if "wandb" in str(config_path):
        cfg = OmegaConf.create(flatten_wandb(OmegaConf.to_container(cfg, resolve=True)))
    return cfg


def _patch_scene_initializer(target_cfg: DictConfig, init_config_path: Path, context: str) -> None:
    """
    Load scene_trainer.scene_initializer from init_config_path and patch it into target_cfg in-place.
    target_cfg must not be struct-protected when this is called.
    """
    init_cfg = _load_checkpoint_cfg(init_config_path)
    initializer_subcfg = OmegaConf.select(init_cfg, "scene_trainer.scene_initializer", default=None)
    if initializer_subcfg is not None:
        print(cyan(f"{context}: patching scene_trainer.scene_initializer from pretrained_initializer config."))
        OmegaConf.update(target_cfg, "scene_trainer.scene_initializer", initializer_subcfg, merge=True)
    else:
        print(cyan("pretrained_initializer config has no scene_trainer.scene_initializer key; skipping patch."))


def _resolve_config_paths(cli_cfg) -> tuple[Path | None, Path | None]:
    """
    Determine which config files to load based on CLI checkpointing settings.

    Returns:
        config_path:             main checkpoint config (optimizer + initializer architecture), or None
        initializer_config_path: separate initializer checkpoint config (overrides main for initializer), or None

    Priority for config_path:
      resume > pretrained_model > pretrained_optimizer (> pretrained_initializer sets initializer_config_path only)
    """
    pretrained_model = cli_cfg.checkpointing.pretrained_model
    pretrained_optimizer = cli_cfg.checkpointing.pretrained_optimizer
    pretrained_initializer = cli_cfg.checkpointing.pretrained_initializer
    should_load = cli_cfg.mode == "test" or cli_cfg.checkpointing.load_existing_cfg

    config_path = None
    initializer_config_path = None

    if pretrained_model is not None:
        if should_load:
            config_path = _find_config_for_checkpoint(pretrained_model)
            print(cyan(f"Loading config from pretrained_model checkpoint {config_path}"
                       if config_path else f"No config found for pretrained_model {pretrained_model}."))

    elif pretrained_optimizer is not None:
        if should_load:
            config_path = _find_config_for_checkpoint(pretrained_optimizer)
            print(cyan(f"Loading config from pretrained_optimizer checkpoint {config_path}"
                       if config_path else f"No config found for pretrained_optimizer {pretrained_optimizer}."))
            if pretrained_initializer is not None:
                initializer_config_path = _find_config_for_checkpoint(pretrained_initializer)
                print(cyan(f"Loading initializer config from pretrained_initializer checkpoint {initializer_config_path}"
                           if initializer_config_path else f"No config found for pretrained_initializer {pretrained_initializer}."))

    elif pretrained_initializer is not None:
        if should_load:
            initializer_config_path = _find_config_for_checkpoint(pretrained_initializer)
            print(cyan(f"Loading initializer-only config from pretrained_initializer checkpoint {initializer_config_path}"
                       if initializer_config_path else f"No config found for pretrained_initializer {pretrained_initializer}."))

    else:
        print(cyan("No pretrained_model, pretrained_optimizer, or pretrained_initializer specified, using cli config only."))

    # Resume overrides config_path to point at the output directory's saved config.
    if cli_cfg.checkpointing.resume and cli_cfg.checkpointing.load_existing_cfg:
        config_path = Path(cli_cfg.output_dir) / "config.yaml"
        print(cyan(f"Resuming: loading config from cfg.output_dir {config_path}"))
    else:
        print(cyan("Not resuming.."))

    if config_path is not None and not config_path.exists():
        print(cyan(f"Config file {config_path} does not exist. Continuing with cli config only."))
        config_path = None
    elif config_path is not None:
        print(cyan(f"Found config file {config_path}."))

    return config_path, initializer_config_path


def _merge_test_mode(
        cli_cfg: DictConfig,
        loaded_cfg: DictConfig,
        initializer_config_path: Path | None,
        pretrained_initializer: str | None,
) -> tuple[DictConfig, DictConfig]:
    """
    Test mode: CLI config is the base for all settings (dataset, test flags, etc.).
    Only optimizer and initializer *architecture* are patched in from checkpoint configs.

    Initializer source priority:
      1. separate initializer_config_path  (pretrained_initializer ckpt with a config file)
      2. main loaded_cfg                   (optimizer checkpoint's bundled initializer)
      3. CLI config as-is                  (pretrained_initializer set but has no config file)

    Returns (merged_cfg, orig_cli_cfg); orig_cli_cfg is the snapshot taken before any
    checkpoint patches so that _apply_cli_overrides can restore explicit CLI values.
    """
    OmegaConf.set_struct(cli_cfg, False)
    # Snapshot BEFORE patching: merged_cfg aliases cli_cfg, so patches below also mutate
    # cli_cfg. _apply_cli_overrides must see the original CLI values, not the patched ones.
    orig_cli_cfg = OmegaConf.create(
        OmegaConf.to_container(cli_cfg, resolve=False, throw_on_missing=False)
    )
    merged_cfg = cli_cfg  # patched in-place

    # Patch optimizer architecture from checkpoint
    optimizer_subcfg = OmegaConf.select(loaded_cfg, "scene_trainer.scene_optimizer", default=None)
    if optimizer_subcfg is not None:
        print(cyan("Test mode: patching scene_trainer.scene_optimizer from checkpoint config."))
        OmegaConf.update(merged_cfg, "scene_trainer.scene_optimizer", optimizer_subcfg, merge=True)

    # Patch initializer architecture (priority order above)
    if initializer_config_path is not None and initializer_config_path.exists():
        _patch_scene_initializer(merged_cfg, initializer_config_path, context="Test mode")
    elif pretrained_initializer is None:
        pass
        # TODO Naama
        # No explicit initializer checkpoint — fall back to the optimizer checkpoint's initializer
        # initializer_subcfg = OmegaConf.select(loaded_cfg, "scene_trainer.scene_initializer", default=None)
        # if initializer_subcfg is not None:
            # print(cyan("Test mode: patching scene_trainer.scene_initializer from checkpoint config."))
            # OmegaConf.update(merged_cfg, "scene_trainer.scene_initializer", initializer_subcfg, merge=True)
    else:
        print(cyan("pretrained_initializer set but has no config file; using CLI scene_initializer config."))

    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg, orig_cli_cfg


def _merge_train_mode(
        cli_cfg: DictConfig,
        loaded_cfg: DictConfig,
        initializer_config_path: Path | None,
) -> tuple[DictConfig, DictConfig]:
    """
    Train mode: checkpoint config takes priority over CLI for all existing fields
    (preserves the trained architecture). CLI fills in any new fields added since training.

    If a separate initializer checkpoint is given, its scene_initializer replaces the one
    inside loaded_cfg before the full merge, so the right initializer architecture is used.

    Returns (merged_cfg, orig_cli_cfg); orig_cli_cfg is the pre-merge snapshot used
    by _apply_cli_overrides to restore explicit CLI values.
    """
    if initializer_config_path is not None and initializer_config_path.exists():
        init_cfg = _load_checkpoint_cfg(initializer_config_path)
        initializer_subcfg = OmegaConf.select(init_cfg, "scene_trainer.scene_initializer", default=None)
        if initializer_subcfg is not None:
            print(cyan("Replacing scene_trainer.scene_initializer in loaded config with initializer config."))
            OmegaConf.update(loaded_cfg, "scene_trainer.scene_initializer", initializer_subcfg, merge=False)
        else:
            print(cyan("pretrained_initializer config has no scene_trainer.scene_initializer key; skipping patch."))

    orig_cli_cfg = OmegaConf.create(
        OmegaConf.to_container(cli_cfg, resolve=False, throw_on_missing=False)
    )
    OmegaConf.set_struct(cli_cfg, False)
    merged_cfg = OmegaConf.merge(cli_cfg, loaded_cfg)  # loaded_cfg wins for existing fields
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg, orig_cli_cfg


def merge_config_from_file(cli_cfg):
    # 1. Determine which config files to load.
    config_path, initializer_config_path = _resolve_config_paths(cli_cfg)

    # 2. No checkpoint config: use CLI as-is, optionally patching in initializer architecture.
    if config_path is None:
        print(cyan(f"No config file found, using cli config only. \n"
                   f"Setting config version to {CURRENT_CFG_VERSION}."))
        cli_cfg["version"] = CURRENT_CFG_VERSION
        if initializer_config_path is not None and initializer_config_path.exists():
            OmegaConf.set_struct(cli_cfg, False)
            _patch_scene_initializer(cli_cfg, initializer_config_path, context="No-checkpoint")
            OmegaConf.set_struct(cli_cfg, True)
        return cli_cfg

    # 3. Load and migrate the checkpoint config.
    print(cyan(f"Loading config from {config_path}."))
    loaded_cfg = _load_checkpoint_cfg(config_path)

    # 4. Merge checkpoint config with CLI config (strategy differs by mode).
    #    Test:  CLI is the base; only optimizer/initializer architecture patched from checkpoint.
    #    Train: checkpoint takes priority; CLI fills in new fields added since training.
    pretrained_initializer = cli_cfg.checkpointing.pretrained_initializer
    if cli_cfg.mode == "test":
        merged_cfg, orig_cli_cfg = _merge_test_mode(
            cli_cfg, loaded_cfg, initializer_config_path, pretrained_initializer
        )
    else:
        merged_cfg, orig_cli_cfg = _merge_train_mode(cli_cfg, loaded_cfg, initializer_config_path)

    # 5. Re-apply CLI overrides so user-specified values win over loaded checkpoint config.
    merged_cfg = _apply_cli_overrides(merged_cfg, orig_cli_cfg, list(HydraConfig.get().overrides.task))

    return merged_cfg


class SkipRun(Exception):
    pass


def setup_output_dir(cfg, cfg_dict):
    if cfg.output_dir != cfg_dict.output_dir:
        if "$" in str(cfg.output_dir):
            # interpolated value, not sure how to make it work.
            cfg.output_dir = CustomPath(cfg_dict.output_dir)
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = CustomPath(
            HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = CustomPath(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    if HydraConfig.get().mode == RunMode.MULTIRUN and output_dir == "placeholder":
        # Hack to overcome multirun issues
        # TODO Naama, need to move to post_init of cfg
        output_dir = CustomPath(hydra.core.hydra_config.HydraConfig.get()["run"]["dir"])
        print(cyan(f"Multirun detected, setting output_dir to {CustomPath(output_dir):link}"))
        # save checkoint path to a file for debugging
        ckpt_path = cfg.checkpointing.pretrained_model or cfg.checkpointing.pretrained_optimizer
        (output_dir / "ckpt_dir.txt").write_text(str(ckpt_path))
    cfg_dict.output_dir = output_dir
    cfg.output_dir = output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    if cfg.mode == 'test':
        if cfg.meta_trainer.test.output_path is None or str(cfg.meta_trainer.test.output_path) in ['placeholder', 'outputs/test']:
            cfg.meta_trainer.test.output_path = output_dir
        if cfg.meta_trainer.test.compute_scores:
            (cfg.meta_trainer.test.output_path / "metrics").mkdir(exist_ok=True, parents=True)
    print(cyan(f"Saving outputs to {CustomPath(output_dir):link}."))

    # Save the config to the output directory.
    cfg_dict_path = output_dir / "config.yaml"

    with open(cfg_dict_path, "w") as f:
        OmegaConf.save(cfg_dict, f)


def get_eval_cfg(cfg_dict):
    if "meta_trainer" in cfg_dict:
        meta_trainer_dict = cfg_dict["meta_trainer"]
    else:
        raise ValueError("No trainer or meta_trainer in cfg_dict")

    if cfg_dict["mode"] == "train" and meta_trainer_dict["train"]["eval_model_every_n_val"] > 0:
        eval_cfg_dict = deepcopy(cfg_dict)
        dataset_dir = str(cfg_dict["dataset"]["roots"]).lower()
        if "re10k" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/evaluation_index_re10k.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 4:
                eval_path = "assets/re10k_start_0_distance_150_ctx_4v_tgt_6v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/re10k_start_0_distance_200_ctx_6v_tgt_6v.json"
            else:
                if meta_trainer_dict["eval_index"] is not None:
                    eval_path = None  # placeholder
                else:
                    raise ValueError("unsupported number of views for re10k")
        elif "dl3dv" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/dl3dv_start_0_distance_20_ctx_2v_tgt_4v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 8:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 16:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_80_ctx_16v_tgt_16v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 32:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_160_ctx_32v_tgt_24v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 64:
                eval_path = "assets/dl3dv_benchmark/dl3dv_ctx_64v_tgt_every8th.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == -1:
                print("Setting manually eval_path, num_context_views remains -1 for dl3dv eval")
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"
            else:
                raise ValueError("unsupported number of views for dl3dv")
        elif "scannet" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/evaluation_index_scannet_view2.json"
            else:
                raise ValueError("unsupported number of views for scannet")
        elif "tartanair" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = 'assets/evaluation_index_tartanair_view2.json'
            else:
                raise ValueError("unsupported number of views for tartanair")
        else:
            raise Exception("Fail to load eval index path")
        eval_cfg_dict["dataset"]["view_sampler"] = {
            "name": "evaluation",
            "index_path": eval_path,
            "num_context_views": cfg_dict["dataset"]["view_sampler"]["num_context_views"],
        }

        # specify eval index
        if meta_trainer_dict["eval_index"] is not None:
            eval_cfg_dict["dataset"]["view_sampler"]["index_path"] = meta_trainer_dict["eval_index"]

        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None
    return eval_cfg
