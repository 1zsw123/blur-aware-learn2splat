from .optimizer import Optimizer, OptimizerCfg
from .optimizer_learn2splat import Learn2SplatOptimizer
from .optimizer_knn_based import KnnBasedOptimizer, KnnBasedOptimizerCfg
from .optimizer_resplat import ResplatOptimizerV1, ResplatOptimizerV2
from .optimizer_adam import AdamOptimizerCfg, AdamOptimizer

SceneOptimizerCfg = KnnBasedOptimizerCfg | AdamOptimizerCfg


def extract_opt_params(cfg):
    opt_params = cfg.__dict__.copy()
    opt_params.pop("enabled", None)
    opt_params.pop("name", None)
    opt_params.pop("steps", None)
    opt_params.pop("scheduler", None)
    opt_params.pop("scheduler_warm_up_ratio", None)
    opt_params.pop("compute_metrics_every", None)
    return opt_params


SCENE_OPTIMIZERS = {
    "none": None,
    "depthsplat": KnnBasedOptimizer,
    "resplat_v1": ResplatOptimizerV1,
    "resplat_v2": ResplatOptimizerV2,
    "clogs": Learn2SplatOptimizer,  # TODO (release): remove
    "l2s": Learn2SplatOptimizer,
    "adam": AdamOptimizer,
}


def get_scene_optimizer(cfg: SceneOptimizerCfg | None) -> Optimizer | None:
    if cfg is None:
        print("Using scene optimizer: None")
        return None
    print(f"Using scene optimizer: {cfg.name}")
    scene_optimizer = SCENE_OPTIMIZERS[cfg.name]
    if scene_optimizer is None:
        return None
    scene_optimizer = scene_optimizer(cfg)
    return scene_optimizer
