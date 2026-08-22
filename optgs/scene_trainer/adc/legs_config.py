from dataclasses import dataclass
from typing import Literal

from optgs.scene_trainer.adc.base import BaseStrategyCfg


@dataclass
class LeGSStrategyCfg(BaseStrategyCfg):
    """Configuration for the exact transplanted LeGS capacity mechanism."""

    name: Literal["legs", "legs_blur"]
    grad_thresh: float = 0.0001
    grad_abs_thresh: float = 0.0002
    state_dim: int = 11
    hidden_dim: int = 64
    reward_delay: int = 50
    rollout_batch_size: int = 2
    ppo_epochs: int = 2
    ppo_chunk_size: int = 500_000
    state_view_count: int = 10
    use_mixed_precision: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    actor_lr_init: float = 1e-3
    actor_lr_final: float = 1e-5
    state_encoder_lr_init: float = 1e-3
    state_encoder_lr_final: float = 1e-5
    prune_lr_init: float = 1e-3
    prune_lr_final: float = 1e-5
    min_opacity_init: float = 0.005
    min_opacity_final: float = 0.1
    reward_normalize: bool = True
    use_prune_estimator: bool = True
    blur_conditioned: bool = False
    blur_feature_dim: int = 7
    blur_quality_weight: float = 1.0
    blur_capacity_weight: float = 1.0
