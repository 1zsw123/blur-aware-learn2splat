import os
from collections import OrderedDict
from typing import Any

import torch

from optgs.misc.io import cyan


# TODO (release): remove ckpt migration
# Optimizer module attributes renamed in the update_/refine_ cleanup pass.
# Maps every old spelling (including the resplat-era "render_error_mv_attn") to the
# current name. Applied per dot-delimited key segment so it works whether checkpoint
# keys are relative to the optimizer ("update_head.0.weight") or prefixed
# ("optimizer.update_head.0.weight"). Idempotent: current names never match an old
# segment, so re-running on an already-migrated checkpoint is a no-op.
_OPTIMIZER_ATTR_RENAMES = {
    "update_proj": "state_proj",
    "update_feature": "error_feature_extractor",
    "update_rgb_error_proj": "rgb_error_proj",
    "update_input_norm": "input_norm",
    "update_module": "point_transformer",
    "update_head_list": "extra_gaussian_heads",
    "update_head": "delta_head",
    "update_error_attn": "error_mv_attn",
    "render_error_mv_attn": "error_mv_attn",  # resplat-era name
}


def _rename_optimizer_attrs(state_dict):
    """Rewrite checkpoint keys for renamed optimizer module attributes (see map above)."""
    renamed = {}
    for k, v in state_dict.items():
        parts = [_OPTIMIZER_ATTR_RENAMES.get(p, p) for p in k.split(".")]
        renamed[".".join(parts)] = v
    return renamed


# Function to extract the step number from the filename
def extract_step(file_name):
    step_str = file_name.split("-")[1].split("_")[1].replace(".ckpt", "")
    return int(step_str)


def find_latest_ckpt(ckpt_dir):
    # List all files in the directory that end with .ckpt
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    # Check if there are any .ckpt files in the directory
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in {ckpt_dir}.")
    else:
        # Find the file with the maximum step
        latest_ckpt_file = max(ckpt_files, key=extract_step)
        return ckpt_dir / latest_ckpt_file


def no_resume_upsampler(pretrained_state_dict):
    new_state_dict = OrderedDict()
    for key, value in pretrained_state_dict.items():
        if 'upsampler' not in key:
            new_state_dict[key] = value
    return new_state_dict


def load_partial_state_dict(model, pretrained_state_dict):
    # Load only matching parameters
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)


def _load_state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    if 'model' in ckpt:
        return ckpt['model']
    return ckpt


def load_optimizer(cfg, scene_trainer, strict_load):
    pretrained_model = torch.load(cfg.checkpointing.pretrained_optimizer, map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']
    # Strip scene_trainer. prefix if present (Lightning checkpoint format)
    pretrained_model = {k.replace("scene_trainer.", ""): v for k, v in pretrained_model.items()}
    if any(k.startswith("optimizer.") for k in pretrained_model):
        # Unified repo format: keys are optimizer.*
        optimizer_state_dict = {k[len("optimizer."):]: v for k, v in pretrained_model.items() if
                                k.startswith("optimizer.")}
    else:
        # Resplat repo format: keys are encoder.* (before init/opt split).
        # Strip encoder. prefix; init-related keys will be ignored via strict=False.
        optimizer_state_dict = {k[len("encoder."):]: v for k, v in pretrained_model.items() if k.startswith("encoder.")}

    # Rename module attributes that were renamed in the update_/refine_ cleanup pass
    optimizer_state_dict = _rename_optimizer_attrs(optimizer_state_dict)

    # If init_state_wo_features is True, remove all feature-related parameters from the optimizer state dict
    if getattr(cfg.scene_trainer.scene_optimizer, "init_state_wo_features", False):
        optimizer_state_dict = {k: v for k, v in optimizer_state_dict.items() if "state_proj" not in k}
    scene_trainer.optimizer.load_state_dict(optimizer_state_dict, strict=strict_load)
    print(cyan(f"Loaded pretrained optimizer: {cfg.checkpointing.pretrained_optimizer}"))


def load_initializer(cfg, scene_trainer, strict_load):
    pretrained_model = torch.load(cfg.checkpointing.pretrained_initializer, map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']
    # Strip scene_trainer. prefix if present (Lightning checkpoint format)
    pretrained_model = {k.replace("scene_trainer.", ""): v for k, v in pretrained_model.items()}
    if any(k.startswith("initializer.") for k in pretrained_model):
        assert all(k.startswith("initializer.") for k in pretrained_model)
        # Current repo format: keys are initializer.*
        initializer_state_dict = {k[len("initializer."):]: v for k, v in pretrained_model.items() if
                                  k.startswith("initializer.")}
    else:
        # Resplat repo format: keys are encoder.* (before init/opt split)
        initializer_state_dict = {k[len("encoder."):]: v for k, v in pretrained_model.items() if
                                  k.startswith("encoder.")}
    scene_trainer.initializer.load_state_dict(initializer_state_dict, strict=strict_load)
    print(cyan(f"Loaded pretrained initializer: {cfg.checkpointing.pretrained_initializer}"))


def load_full_model(cfg, scene_trainer, strict_load):
    pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']
    # Rename optimizer module attributes renamed in the update_/refine_ cleanup pass.
    pretrained_model = _rename_optimizer_attrs(pretrained_model)
    if cfg.checkpointing.partial_load:
        print('partial load')
        load_partial_state_dict(scene_trainer, pretrained_model)
    else:
        scene_trainer.load_state_dict(pretrained_model, strict=strict_load)
    print(cyan(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"))


def load_base_model(cfg, scene_trainer, strict_load: bool | Any):
    if cfg.checkpointing.pretrained_model is not None:
        load_full_model(cfg, scene_trainer, strict_load)
    else:
        # Load pretrained initializer if available
        if cfg.checkpointing.pretrained_initializer is not None:
            load_initializer(cfg, scene_trainer, strict_load)

        if cfg.checkpointing.pretrained_optimizer is not None and scene_trainer.optimizer is not None:
            load_optimizer(cfg, scene_trainer, strict_load)


def load_model_weights(cfg, scene_trainer, strict_load, mode: str):
    assert mode in ("train", "test")

    if mode == "train":
        # only load monodepth
        if cfg.checkpointing.pretrained_monodepth is not None:
            strict_load = False
            pretrained_model = torch.load(cfg.checkpointing.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']
            if cfg.model.encoder.separate_depth_color or cfg.model.encoder.separate_depth_gaussian_scale:
                scene_trainer.encoder.feature_extractor.load_state_dict(pretrained_model, strict=strict_load)
            else:
                scene_trainer.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(cyan(f"Loaded pretrained monodepth: {cfg.checkpointing.pretrained_monodepth}"))

        # freeze mono vit
        if cfg.checkpointing.freeze_mono_vit:
            print('freeze mono vit')
            for params in scene_trainer.encoder.depth_predictor.pretrained.parameters():
                params.requires_grad = False

        # load pretrained mvdepth
        if cfg.checkpointing.pretrained_mvdepth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_mvdepth, map_location='cpu')['model']
            if cfg.model.encoder.separate_depth_color or cfg.model.encoder.separate_depth_gaussian_scale:
                scene_trainer.encoder.feature_extractor.load_state_dict(pretrained_model, strict=False)
            else:
                scene_trainer.encoder.depth_predictor.load_state_dict(pretrained_model, strict=False)
            print(cyan(f"Loaded pretrained mvdepth: {cfg.checkpointing.pretrained_mvdepth}"))

    # load full model (or separate initializer/optimizer checkpoints)
    load_base_model(cfg, scene_trainer, strict_load)

    # load pretrained depth
    if cfg.checkpointing.pretrained_depth is not None:
        pretrained_model = _load_state_dict(cfg.checkpointing.pretrained_depth)
        if mode == "train":
            if cfg.checkpointing.partial_load:
                print('partial load depth')
                load_partial_state_dict(scene_trainer.initializer.depth_predictor, pretrained_model)
            else:
                if cfg.checkpointing.no_resume_upsampler:
                    pretrained_model = no_resume_upsampler(pretrained_model)
                    strict_load = False
                scene_trainer.initializer.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
        else:
            scene_trainer.initializer.depth_predictor.load_state_dict(pretrained_model, strict=True)
        print(cyan(f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"))

    # load pretrained scale predictor
    if mode == "train" and cfg.checkpointing.pretrained_scale_predictor is not None:
        pretrained_model = _load_state_dict(cfg.checkpointing.pretrained_scale_predictor)
        scene_trainer.encoder.scale_predictor.load_state_dict(pretrained_model, strict=strict_load)
        print(cyan(f"Loaded pretrained scale predictor: {cfg.checkpointing.pretrained_scale_predictor}"))

        print('freeze scale predictor')
        for params in scene_trainer.encoder.scale_predictor.parameters():
            params.requires_grad = False

    # load pretrained update module
    if cfg.checkpointing.resume_update_module is not None:
        pretrained_model = _load_state_dict(cfg.checkpointing.resume_update_module)

        # Filter and load only matching "update_" parameters
        filtered_dict = {
            k: v for k, v in pretrained_model.items()
            if "encoder.update" in k and k in scene_trainer.state_dict()
            and v.shape == scene_trainer.state_dict()[k].shape
        }

        # Load them using strict=False so it skips missing/unmatched keys
        scene_trainer.load_state_dict(filtered_dict, strict=False)
        print(cyan(f"Loaded pretrained update module: {cfg.checkpointing.resume_update_module}"))

    if mode == "train":
        apply_freezes(cfg, scene_trainer)


def apply_freezes(cfg, scene_trainer):
    if getattr(cfg.scene_trainer.scene_initializer, 'freeze_depth', False):
        print('freeze depth')
        for params in scene_trainer.initializer.depth_predictor.parameters():
            params.requires_grad = False

    if not cfg.scene_trainer.train_scene_init:
        print('train refine only, freezing scene initializer')
        for name, params in scene_trainer.initializer.named_parameters():
            params.requires_grad = False

    if cfg.scene_trainer.num_update_steps > 0:
        if not cfg.scene_trainer.train_scene_opt:
            print('train refine only, freezing scene optimizer')
            for name, params in scene_trainer.optimizer.named_parameters():
                params.requires_grad = False
        if cfg.scene_trainer.scene_optimizer.train_global_only:
            print('train global update only')
            for name, params in scene_trainer.optimizer.named_parameters():
                if 'global_update' not in name:
                    params.requires_grad = False