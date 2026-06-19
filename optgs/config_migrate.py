from math import log

from omegaconf import OmegaConf


CURRENT_CFG_VERSION = 2.5

def migrate(cfg_dict):
    was_omega = not isinstance(cfg_dict, dict)
    version = cfg_dict.get("version", 0)

    # null means a fresh run from main.yaml — treat as current version.
    if version is None:
        version = CURRENT_CFG_VERSION

    if version == 0:
        # Heuristic: configs that were partially migrated may have version=0 but a
        # non-depthsplat optimizer name (already renamed during v0→v1), so skip v0→v1.
        so = cfg_dict.get("scene_trainer", {}).get("scene_optimizer", {})
        if so.get("name", "") not in ["depthsplat"]:
            version = 1
        else:
            print("Migrating config from version 0 (cvpr submission) to version 1 (cvpr rebuttal)...")
            cfg_dict = migrate_v0_to_v1(cfg_dict)
            version = 1

    if version == 1:
        print("Migrating config from version 1 to version 2 (train/test moved under meta_trainer)...")
        cfg_dict = migrate_v1_to_v2(cfg_dict)
        version = 2

    if version == 2:
        print("Migrating config from version 2 to version 2.1 (strip update_/refine_ prefixes)...")
        cfg_dict = migrate_v2_to_v2_1(cfg_dict)
        version = 2.1

    if version == 2.1:
        print("Migrating config from version 2.1 to version 2.2 (single-space scale clamp)...")
        cfg_dict = migrate_v2_1_to_v2_2(cfg_dict)
        version = 2.2

    if version == 2.2:
        print("Migrating config from version 2.2 to version 2.3 (replay_buffer -> ckpt_buffer, simulate_ahead -> rollout)...")
        cfg_dict = migrate_v2_2_to_v2_3(cfg_dict)
        version = 2.3

    if version == 2.3:
        print("Migrating config from version 2.3 to version 2.4 (adam flat LRs -> expon lr_scheduler)...")
        cfg_dict = migrate_v2_3_to_v2_4(cfg_dict)
        version = 2.4

    if version == 2.4:
        print("Migrating config from version 2.4 to version 2.5 (meta_optimizer moved under meta_trainer)...")
        cfg_dict = migrate_v2_4_to_v2_5(cfg_dict)
        version = 2.5

    if version != CURRENT_CFG_VERSION:
        raise ValueError(f"Unsupported config version: {version}")

    # Apply code-level renames and strip stale fields.
    # Work on a plain dict so mutations propagate; convert back to OmegaConf if needed.
    cfg_container = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else cfg_dict

    # Handle code-level renames that don't require a version bump (e.g. resplat → resplat_v1).
    so = cfg_container.get("scene_trainer", {}).get("scene_optimizer", {})
    si = cfg_container.get("scene_trainer", {}).get("scene_initializer", {})
    if so.get("name") == "resplat":
        so["name"] = "resplat_v1"
    if si.get("name") == "resplat":
        si["name"] = "resplat_v1"

    # Strip stale postprocessing fields from old checkpoint configs
    te = cfg_container.get("meta_trainer", {}).get("test", {})
    pp = te.get("postprocessing", None) if isinstance(te, dict) else None
    if isinstance(pp, dict):
        pp.pop("__target__", None)
        pp.pop("enabled", None)
        pp.pop("lr", None)

    # save_point_cloud removed (the flag was never read by any code path).
    if isinstance(te, dict):
        te.pop("save_point_cloud", None)

    # Video flags renamed (fixed_view->optim, fixed_iteration->orbit, combined->optim_orbit).
    # Test flags come from the current run's config, so just drop the stale keys from old configs.
    for k in ("save_video_fixed_view", "save_video_fixed_view_index", "save_video_fixed_view_duplicate",
              "save_video_fixed_iteration", "save_video_fixed_iteration_indices",
              "save_video_fixed_iteration_render_fixed_view", "save_video_combined",
              "save_video_combined_iterations", "save_video_combined_fixed_iteration_length"):
        if isinstance(te, dict):
            te.pop(k, None)

    # Strip stale foundationstereo fields (encoder removed)
    si.pop("foundationstereo", None)
    si.pop("fstereo_num_refine", None)

    # Strip removed optimizer sliding-window fields (feature removed)
    for k in ("window_size", "update_window_size", "local_gaussian_render",
              "window_local_refine", "window_global_refine", "window_local_global_refine"):
        so.pop(k, None)

    # delta_head_scale_mag removed (experimental). Drop it under all historical names.
    for k in ("delta_head_scale_mag", "update_head_scale_mag", "refine_output_scale_mag"):
        so.pop(k, None)

    # reinit_gaussian_when_multiple removed (reinit path was never implemented). Drop it
    # under all historical names (reinit_gaussian_when_refine_multiple is the pre-2.1 name).
    for k in ("reinit_gaussian_when_multiple", "reinit_gaussian_when_refine_multiple"):
        so.pop(k, None)

    # The initializer has no point-downsampling path, so pt_downsample,
    # fps_agg_func and subsample_method are not valid config keys; drop them.
    # multi_scale_pt and fps_num_samples are likewise not config keys (the point
    # transformer is always PlainPointTransformer). subsample_method also appears
    # on old optimizer configs, so strip it from there too.
    for k in ("multi_scale_pt", "refine_multi_scale_pt", "subsample_method"):
        so.pop(k, None)
    for k in ("multi_scale_pt", "fps_num_samples", "pt_downsample", "fps_agg_func", "subsample_method"):
        si.pop(k, None)

    # The lvsm gaussian-regressor path was removed (disabled in every config), so
    # lvsm_gaussian_regressor and lvsm_layers are no longer valid initializer keys.
    for k in ("lvsm_gaussian_regressor", "lvsm_layers"):
        si.pop(k, None)

    # pt_pred_residual_position removed (residual-position prediction was disabled in every config).
    si.pop("pt_pred_residual_position", None)

    # sh_only removed: it was "freeze everything but SH", now expressed via per-group freeze_*.
    # (refine_sh_only is the pre-2.1 name.)
    sh_only = so.pop("sh_only", None)
    if sh_only is None:
        sh_only = so.pop("refine_sh_only", None)
    if sh_only:
        for fk in ("freeze_mean", "freeze_scale", "freeze_rotation", "freeze_opacity"):
            so[fk] = True

    # l1_loss / train_ignore_large_loss / half_res_lpips_loss moved off train into the per-loss
    # cfgs: mse/sgd take l1_loss + clamp_large_error, lpips takes half_res.
    train = cfg_container.get("meta_trainer", {}).get("train", {})

    # depth_range_from_disparity removed (disabled in every config); its only consumers,
    # max_disparity and min_disparity, are removed with it.
    for k in ("depth_range_from_disparity", "max_disparity", "min_disparity"):
        train.pop(k, None)

    old_l1 = train.pop("l1_loss", None)
    old_clamp = train.pop("train_ignore_large_loss", None)
    old_half_res = train.pop("half_res_lpips_loss", None)
    for loss_wrapper in cfg_container.get("loss", []) or []:
        if not isinstance(loss_wrapper, dict):
            continue
        for name, lcfg in loss_wrapper.items():
            if not isinstance(lcfg, dict):
                continue
            if name in ("mse", "sgd"):
                lcfg.setdefault("l1_loss", old_l1 if old_l1 is not None else False)
                lcfg.setdefault("clamp_large_error", old_clamp if old_clamp is not None else 0.0)
            elif name == "lpips":
                lcfg.setdefault("half_res", old_half_res if old_half_res is not None else False)

    if was_omega:
        return OmegaConf.create(cfg_container)
    return cfg_container


def migrate_v1_to_v2(cfg_dict):
    """
    Migration from v1 to v2: move top-level 'train' and 'test' under 'meta_trainer'.
    """
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    meta_trainer = cfg.setdefault("meta_trainer", {})

    for key in ("train", "test"):
        if key in cfg and key not in meta_trainer:
            meta_trainer[key] = cfg.pop(key)

    cfg["version"] = 2
    return cfg


def migrate_v2_4_to_v2_5(cfg_dict):
    """Migration from v2.4 to v2.5: move top-level 'meta_optimizer' under 'meta_trainer'."""
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    if "meta_optimizer" in cfg:
        meta_trainer = cfg.setdefault("meta_trainer", {})
        if "meta_optimizer" not in meta_trainer:
            meta_trainer["meta_optimizer"] = cfg.pop("meta_optimizer")

    cfg["version"] = 2.5
    return cfg


def migrate_v2_to_v2_1(cfg_dict):
    """
    Migration from v2 to v2.1: strip redundant update_/refine_ prefixes from optimizer
    config fields (the class is already named Optimizer), rename no_refine_* -> freeze_*,
    and update_head_* -> delta_head_*.
    """
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    so = cfg.get("scene_trainer", {}).get("scene_optimizer", {})

    RENAME_MAP = {
        # refine_
        "num_basic_refine_blocks": "num_basic_blocks",
        "num_refine_blocks": "num_blocks",
        "refine_block_rmsnorm": "block_rmsnorm",
        "refine_block_layernorm": "block_layernorm",
        "refine_gaussian_multiple": "delta_gaussian_multiple",
        "refine_residual_init_state": "residual_init_state",
        "clamp_refine_max_scale": "clamp_max_scale",
        "refine_condition_pt_feature": "condition_pt_feature",
        "refine_same_num_points": "same_num_points",
        "refine_knn_samples": "knn_samples",
        "refine_with_mv_attn": "with_mv_attn",
        "refine_with_mv_attn_lowres": "with_mv_attn_lowres",
        "refine_no_mv_attn": "no_mv_attn",
        "refine_mv_shuffle_attn": "mv_shuffle_attn",
        "refine_mv_attn_with_pos_enc": "mv_attn_with_pos_enc",
        "refine_shuffle_attn_no_norm": "shuffle_attn_no_norm",
        "refine_mv_unimatch_attn": "mv_unimatch_attn",
        # no_refine_ -> freeze_
        "no_refine_mean": "freeze_mean",
        "no_refine_scale": "freeze_scale",
        "no_refine_rotation": "freeze_rotation",
        "no_refine_opacity": "freeze_opacity",
        "no_refine_sh0": "freeze_sh0",
        "no_refine_shN": "freeze_shN",
        # other update_
        "update_attn_proj_channels": "attn_proj_channels",
        "update_no_knn_attn": "no_knn_attn",
        "update_no_tran_block_norm": "no_tran_block_norm",
        "update_tran_block_act": "tran_block_act",
        "train_global_update_only": "train_global_only",
        "random_update_with_size": "random_step_with_size",
        "gradient_update_scale": "residual_grad_scale",
        # update_head_ -> delta_head_
        "update_head_layer_num": "delta_head_layer_num",
        "update_head_concat_img": "delta_head_concat_img",
        "update_head_act": "delta_head_act",
        "update_head_final_act": "delta_head_final_act",
        "update_head_hidden_dim_matches": "delta_head_hidden_dim_matches",
        "update_head_scalar_scale": "delta_head_scalar_scale",
        "update_head_scalar_scale_act": "delta_head_scalar_scale_act",
        "update_head_per_param_heads": "delta_head_per_param_heads",
        "update_head_per_param_hidden_dim": "delta_head_per_param_hidden_dim",
        "update_head_per_param_scales": "delta_head_per_param_scales",
    }

    for old, new in RENAME_MAP.items():
        if old in so:
            so[new] = so.pop(old)

    cfg["version"] = 2.1
    return cfg


def migrate_v2_1_to_v2_2(cfg_dict):
    """
    Migration from v2.1 to v2.2: scales are now clamped in a single space.

    When opt_scales_before_act is set, scales are refined in log space, and the clamp is now
    applied there (raw limits) both when unpacking and in the per-step update. Earlier configs
    instead clamped once in activation space (clamp_min/max_scale) during unpack and once in log
    space (clamp_min/max_raw_scales) during the update, so the scale was effectively bounded by
    the intersection of the two. Fold that intersection into the raw limits so old checkpoints
    keep the same effective bounds under the new single-space clamp.
    """
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    so = cfg.get("scene_trainer", {}).get("scene_optimizer", {})
    if so.get("opt_scales_before_act", False):
        min_scale = float(so.get("clamp_min_scale", 1e-6))
        max_scale = float(so.get("clamp_max_scale", 3.0))
        min_raw = float(so.get("clamp_min_raw_scales", -1e10))
        max_raw = float(so.get("clamp_max_raw_scales", 1e10))
        so["clamp_min_raw_scales"] = max(log(min_scale), min_raw)
        so["clamp_max_raw_scales"] = min(log(max_scale), max_raw)

    cfg["version"] = 2.2
    return cfg


def migrate_v2_2_to_v2_3(cfg_dict):
    """
    Migration from v2.2 to v2.3: the replay-buffer feature is renamed to ckpt_buffer and
    simulate_ahead to rollout (use_replay_buffer -> use_ckpt_buffer, replay_buffer_cfg ->
    ckpt_buffer_cfg, and the simulate_ahead* fields inside it -> rollout*).
    """
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    train = cfg.get("meta_trainer", {}).get("train", {})
    if "use_replay_buffer" in train:
        train["use_ckpt_buffer"] = train.pop("use_replay_buffer")
    if "replay_buffer_cfg" in train:
        train["ckpt_buffer_cfg"] = train.pop("replay_buffer_cfg")

    buffer_cfg = train.get("ckpt_buffer_cfg", {})
    if isinstance(buffer_cfg, dict):
        ROLLOUT_RENAME_MAP = {
            "simulate_ahead": "rollout",
            "simulate_ahead_min_steps": "rollout_min_steps",
            "simulate_ahead_max_steps": "rollout_max_steps",
            "simulate_ahead_grow": "rollout_grow",
        }
        for old, new in ROLLOUT_RENAME_MAP.items():
            if old in buffer_cfg:
                buffer_cfg[new] = buffer_cfg.pop(old)

    cfg["version"] = 2.3
    return cfg


def migrate_v2_3_to_v2_4(cfg_dict):
    """
    Migration from v2.3 to v2.4: the Adam baseline (name adam/sgd) no longer carries flat per-param
    LR fields. They move into the inherited lr_scheduler cfg (name "expon"): per-param values into
    lr_data, the means decay into the expon scheduler params, with apply_scheduler enabling the
    schedule on means only.

    The expon scheduler scales the whole curve by lr_data._base (= old base_lr) and the means LR is
    additionally scaled by scene extent at runtime, so the bounds reproduce the old behavior:
    old means LR = base_lr * scene_scale * expon(means_lr_init -> means_lr_final).
    """
    cfg = OmegaConf.to_container(cfg_dict, resolve=False) if not isinstance(cfg_dict, dict) else dict(cfg_dict)

    so = cfg.get("scene_trainer", {}).get("scene_optimizer", {})
    if so.get("name") in ("adam", "sgd") and "means_lr_init" in so:
        base = so.pop("base_lr", 1)
        lr_scheduler = so.get("lr_scheduler", {}) or {}
        lr_scheduler["name"] = "expon"
        lr_scheduler["lr_data"] = {
            "_base": base,
            "_means": so.pop("means_lr_init"),
            "_scales": so.pop("scales_lr"),
            "_quats": so.pop("rotations_lr"),
            "_opacities": so.pop("opacities_lr"),
            "_sh0": so.pop("sh0s_lr"),
            "_shN": so.pop("shNs_lr"),
        }
        # _base gates all params (Bool3DGSCfg.<param> = _base AND _param), so it must stay True;
        # the schedule is enabled per-param via the individual flags (means only).
        lr_scheduler["apply_scheduler"] = {
            "_base": True, "_means": True, "_scales": False,
            "_quats": False, "_opacities": False, "_sh0": False, "_shN": False,
        }
        lr_scheduler["lr_final"] = so.pop("means_lr_final")
        lr_scheduler["lr_delay_steps"] = 0
        lr_scheduler["lr_delay_mult"] = so.pop("means_lr_delay_mult", 1.0)
        lr_scheduler["max_steps"] = so.pop("means_lr_max_steps", 30000)
        so["lr_scheduler"] = lr_scheduler

    cfg["version"] = 2.4
    return cfg


def migrate_v0_to_v1(cfg):
    """
    Migration from submission v0 (refine_*) to rebuttal v1 (input_error_*).
    """

    cfg = OmegaConf.to_container(cfg, resolve=False)

    so = cfg["scene_trainer"]["scene_optimizer"]
    si = cfg["scene_trainer"]["scene_initializer"]

    # ------------------------------------------------------------------
    # Module renames
    # ------------------------------------------------------------------
    if si["name"] == "depthsplat":
        si["name"] = "resplat_v1"
    if so["name"] == "depthsplat":
        if so["refine_input_gradient"]:
            so["name"] = "learn2splat"
        else:
            so["name"] = "resplat_v1"

    # ------------------------------------------------------------------
    # Key renames (declarative)
    # ------------------------------------------------------------------
    RENAME_MAP = {
        # feature extraction
        "refine_lpips_error": "input_error_lpips_features",
        "refine_pool_vgg_features": "input_error_pool_vgg_features",
        "refine_use_all_vgg_features": "input_error_use_all_vgg_features",
        "refine_vit_feature": "input_error_vit_feature",
        "refine_resnet_feature": "input_error_resnet_feature",
        "no_freeze_resnet_feature": "input_error_no_freeze_resnet_feature",
        "shallow_resnet_feature": "input_error_shallow_resnet_feature",
        "resnet_feature_layers": "input_error_resnet_feature_layers",
        "refine_convnext_feature": "input_error_convnext_feature",
        "convnext_feature_size": "input_error_convnext_feature_size",
        "refine_concat_feature": "input_error_concat_feature",
        "refine_concat_feature_cosine": "input_error_concat_feature_cosine",
        "refine_cosine_feature": "input_error_cosine_feature",
        "refine_add_feature": "input_error_add_feature",
        "refine_concat_rgb_feature_error": "input_error_concat_rgb_feature_error",

        # render error → input error
        "render_error_no_abs": "input_error_no_abs",
        "render_error_no_shuffle": "input_error_no_shuffle",
        "render_cache_resnet_feature": "input_error_cache_resnet_feature",
        "render_view_pool_resnet_feature": "input_error_view_pool_resnet_feature",
        "render_global_pool_resnet_feature": "input_error_global_pool_resnet_feature",

        # input toggles
        "refine_input_alpha": "input_alpha",
        "refine_input_depth": "input_depth",
        "refine_input_depth_smooth_error": "input_depth_smooth_error",
        "refine_input_error": "input_error",

        # attention (input error)
        "radii_averaged_render_error": "input_error_radii_averaged",
        "cross_attn_additional_render_error": "input_error_additional_cross_attn",
        "num_intermediate_views": "input_error_num_intermediate_views",
        "render_error_mv_attn_blocks": "input_error_mv_attn_blocks",

        # context handling
        "render_error_num_views": "input_error_num_views",
        "render_error_remain_context": "input_error_remain_context",
        "render_error_merge_remain_context": "input_error_merge_remain_context",
        "render_error_warp_remain_context": "input_error_warp_remain_context",
        "render_error_random_num_remain_context": "input_error_random_num_remain_context",
        "render_error_num_remain_context_test": "input_error_num_remain_context_test",
        "render_error_warp_input_view": "input_error_warp_input_view",

        # input gradient
        "refine_input_gradient": "input_gradient",
        "refine_input_gradient_log": "input_gradient_log",
        "refine_input_gradient_log_clip_deltas": "input_gradient_log_clip_deltas",
        "refine_input_gradient_scale": "input_gradient_scale",

        # normalize input
        "normalize_update_input": "input_gradient_normalize",
        "normalize_update_input_type": "input_gradient_normalize_type",
        "normalize_state": "input_normalize_state",
        "normalize_gaussians": "input_normalize_gaussians",


        # update head
        "final_head_act": "update_head_final_act",
        "scalar_scale_out": "update_head_scalar_scale",
        "scalar_scale_out_act": "update_head_scalar_scale_act",

    }

    for old, new in RENAME_MAP.items():
        if old in so:
            so[new] = so.pop(old)

    # ------------------------------------------------------------------
    # New / fixed defaults
    # ------------------------------------------------------------------
    if so["name"] in ["clogs", "learn2splat", "resplat_v1"]:
        so["update_head_hidden_dim_matches"] = "output"
    else:
        raise NotImplementedError

    if so["state_channels"] == 0:
        so["state_channels"] = 256

    # ------------------------------------------------------------------
    # Version bump
    # ------------------------------------------------------------------
    cfg["version"] = 1

    return OmegaConf.create(cfg)
