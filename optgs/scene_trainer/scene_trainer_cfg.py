from dataclasses import dataclass
from typing import Literal

from .initializer import InitializerCfg
from .optimizer import SceneOptimizerCfg
from ..model.decoder import DecoderCfg


@dataclass
class SceneTrainerCfg:
    scene_initializer: InitializerCfg
    scene_optimizer: SceneOptimizerCfg | None
    decoder: DecoderCfg
    use_fsdp: bool
    train_scene_init: bool
    train_scene_opt: bool
    num_update_steps: int
    iter_batch_size: int  # if -1, use full batch

    train_min_refine: int
    train_max_refine: int

    opt_batch_size: int  # if -1, use full batch
    opt_batch_size_min: int  # if > 0, use random sub-batch
    opt_batch_size_max: int  # if > 0, use random sub-batch
    opt_batch_strategy: Literal["random", "sequential", "neighbors", "fps"]  # strategy for sub-batch
    sh_degree_interval: int  # 0 = disabled; N = steps between SH degree increments (like gsplat simple_trainer)

    def __post_init__(self):
        if self.scene_optimizer is not None:
            self.scene_optimizer.update(self.scene_initializer)
