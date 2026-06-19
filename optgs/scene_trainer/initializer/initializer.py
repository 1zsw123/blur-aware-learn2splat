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
    # View indices used when rendering a subset (training); None means all views were rendered.
    target_render_index: torch.Tensor | None = None
    context_render_index: torch.Tensor | None = None

    def get_render(self, which: str) -> DecoderOutput | None:
        if which == "target":
            return self.target_render
        elif which == "context":
            return self.context_render
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render(self, which: str, value: DecoderOutput) -> None:
        if which == "target":
            self.target_render = value
        elif which == "context":
            self.context_render = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def get_render_index(self, which: str) -> torch.Tensor | None:
        if which == "target":
            return self.target_render_index
        elif which == "context":
            return self.context_render_index
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render_index(self, which: str, value: torch.Tensor | None) -> None:
        if which == "target":
            self.target_render_index = value
        elif which == "context":
            self.context_render_index = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")


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

    def get_gaussian_param_num(self) -> int:
        """Per-Gaussian parameter count of this initializer's own representation:
        scale(3) + rotation(4) + SH(3*sh_d) + opacity(1) + position encoding (get_position_param_num)."""
        return 3 + 4 + 3 * self.get_sh_d() + 1 + self.get_position_param_num()

    def get_position_param_num(self) -> int:
        """Size of this initializer's position encoding. Most initializers place Gaussians at explicit
        3D positions (3). Initializers that encode position differently (resplat: a 2D pixel offset
        added to a per-pixel depth) override this."""
        return 3

@dataclass
class NonlearnedInitializerCfg(InitializerCfg):
    pass

@dataclass
class LearnedInitializerCfg(InitializerCfg):
    pass


@dataclass
class PerPixelInitializerCfg(LearnedInitializerCfg):
    latent_gs: bool
    latent_downsample: int


class Initializer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def eval_preprocessing(self, batch, train_cfg) -> None:
        """Eval/validation-only batch prep (depth-range override + optional scale prediction),
        applied in-place before the initializer runs. Training does not call this. The
        universal patch-crop data shim is a separate, always-applied step (see MetaTrainer)."""
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
