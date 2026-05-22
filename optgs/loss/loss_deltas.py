from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from optgs.loss import Loss
from optgs.model.decoder.decoder import DecoderOutput
from optgs.model.types import Gaussians
from optgs.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossDeltasCfg:
    weight: float | int
    exclude_by_norm_grad: bool
    exclude_by_norm_grad_opposite: bool
    eps: float
    apply_after_step: int

@dataclass
class LossDeltasCfgWrapper:
    deltas: LossDeltasCfg


class LossDeltas(Loss[LossDeltasCfg, LossDeltasCfgWrapper]):
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

        cfg = self.cfg
        # Before the specified step, don't apply the loss.
        if global_step < cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=prediction.color.device)

        if gaussians is None:
            raise ValueError("Gaussians must be provided for LossDeltas.")        

        predicted_deltas = gaussians.deltas

        if not cfg.exclude_by_norm_grad:
            return predicted_deltas.abs().mean() * cfg.weight

        norm_g = gaussians.norm_gradients
        if norm_g is None:
            return predicted_deltas.abs().mean() * cfg.weight

        g = gaussians.gradients
        eps = cfg.eps
        g_abs = g.abs()

        # Condition 1: small gradients
        cond_small = g_abs < eps
        mask = cond_small

        # Condition 2: large gradients but opposite sign
        # deltas are added (sgd substract), so in practice we want to exclude when they have the same sign
        if cfg.exclude_by_norm_grad_opposite:
            cond_opposite = (g_abs > self.cfg.eps) & (norm_g.sign() == predicted_deltas.sign())
            # Combine both
            mask = cond_small | cond_opposite

        if not mask.any():
            return prediction.color.new_zeros((), dtype=torch.float32)

        # predicted_deltas[mask] creates a new tensor
        # return predicted_deltas[mask].abs().mean() * cfg.weight
        # alternative without indexing
        mask_f = mask.to(predicted_deltas.dtype)

        loss = (predicted_deltas.abs() * mask_f).sum() / mask_f.sum()
        return loss * cfg.weight
