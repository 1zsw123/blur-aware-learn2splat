from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Generic

import torch
from torch import nn

from optgs.model.types import Gaussians
from optgs.model.decoder.decoder import DecoderOutput

T = TypeVar("T")


@dataclass
class InitializerOutput:
    gaussians: Gaussians
    features: torch.Tensor | None = None
    depths: list[torch.Tensor] | torch.Tensor | None = None
    target_render: DecoderOutput | None = None
    context_render: DecoderOutput | None = None


@dataclass
class InitializerCfg:
    per_pixel: bool
    per_view: bool

    # Gaussian subsampling augmentation (applied before fixed_gaussians_num)
    # Set min=max for a fixed subsample count, or use floats for ratio-based sampling
    train_min_gaussians_subsample: int | float | None
    train_max_gaussians_subsample: int | float | None
    eval_min_gaussians_subsample: int | float | None
    eval_max_gaussians_subsample: int | float | None

    # Final fixed Gaussian count for DDP consistency (subsample or pad to reach this)
    # Applied after subsampling augmentation
    train_fixed_gaussians_num: int | None
    eval_fixed_gaussians_num: int | None

@dataclass
class NonlearnedInitializerCfg(InitializerCfg):
    pass

@dataclass
class LearnedInitializerCfg(InitializerCfg):
    pass


@dataclass
class PerPixelInitializerCfg(InitializerCfg):
    latent_gs: bool
    latent_downsample: int


class Initializer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def preprocessing(self, batch, train_cfg) -> None:
        pass

    @property
    def strategy(self) -> str:
        raise NotImplementedError()


class LearnedInitializer(Initializer[T], ABC):
    @property
    def strategy(self) -> str:
        return "learned"


class NonlearnedInitializer(Initializer[T], ABC):
    @property
    def strategy(self) -> str:
        return "nonlearned"
