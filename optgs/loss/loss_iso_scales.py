from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from optgs.loss import Loss
from optgs.model.decoder.decoder import DecoderOutput
from optgs.model.types import Gaussians
from optgs.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossIsoScalesCfg:
    weight: float | int

@dataclass
class LossIsoScalesCfgWrapper:
    iso_scales: LossIsoScalesCfg

class LossIsoScales(Loss[LossIsoScalesCfg, LossIsoScalesCfgWrapper]):
    """ Enforce isotropic scales of the gaussians. """
    def forward(
        self,
        prediction: DecoderOutput,
        gaussians: Gaussians | GaussiansModule | None,
        global_step: int,
        gt_rgb: Tensor,
        pred_rgb: Tensor,
        valid_depth_mask: Tensor | None,
        **kwargs,
    ) -> Float[Tensor, ""]:

        scales = gaussians.scales  # [B, G, 3]
        min_scales = scales.min(-1).values  # [B, G]
        max_scales = scales.max(-1).values  # [B, G]
        aspect_ratio = min_scales / max_scales
        iso_loss = ((aspect_ratio - 1) ** 2).mean()
        return iso_loss * self.cfg.weight


