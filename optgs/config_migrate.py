from omegaconf import OmegaConf


CURRENT_CFG_VERSION = 2

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
    pp = cfg_container.get("meta_trainer", {}).get("test", {}).get("postprocessing", None)
    if isinstance(pp, dict):
        pp.pop("__target__", None)
        pp.pop("enabled", None)
        pp.pop("lr", None)

    # Strip stale foundationstereo fields (encoder removed)
    si.pop("foundationstereo", None)
    si.pop("fstereo_num_refine", None)

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
        "refine_output_scale_mag": "update_head_scale_mag",
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
