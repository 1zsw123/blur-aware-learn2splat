from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from optgs.scene_trainer.gaussian_module import GaussiansModule
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
            self,
            prediction: DecoderOutput,
            gaussians: Gaussians | GaussiansModule | None,
            global_step: int,
            gt_rgb: Tensor,
            pred_rgb: Tensor,
            valid_depth_mask: Tensor | None,
            l1_loss: bool,
            clamp_large_error: float,
            **kwargs,
    ) -> Float[Tensor, ""]:

        error = pred_rgb - gt_rgb  # [B, V, C, H, W]

        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5 and valid_depth_mask.min() < 0.5:
            error = error[~valid_depth_mask]

        if l1_loss:
            # l1 loss
            error = error.abs()
        else:
            # l2 loss
            error = error ** 2

        if clamp_large_error > 0:
            valid_mask = error < clamp_large_error
            error = error[valid_mask]

        error = error.mean()

        return self.cfg.weight * error
