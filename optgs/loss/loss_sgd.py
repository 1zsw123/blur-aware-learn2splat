from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from optgs.dataset.data_types import BatchedExample
from optgs.loss import Loss
from optgs.model.decoder.decoder import DecoderOutput
from optgs.model.types import Gaussians
from optgs.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossSGDCfg:
    pass

@dataclass
class LossSGDCfgWrapper:
    sgd: LossSGDCfg


class LossSGD(Loss[LossSGDCfg, LossSGDCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | GaussiansModule | None,
        global_step: int,
        l1_loss: bool,
        clamp_large_error: float,
        valid_depth_mask: Tensor | None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        
        if gaussians is None:
            raise ValueError("Gaussians must be provided for LossDeltas.")        

        predicted_deltas = gaussians.deltas
        gt_gradients = gaussians.gradients

        # cast to float16 if necessary
        if predicted_deltas.dtype != gt_gradients.dtype:
            gt_gradients = gt_gradients.to(predicted_deltas.dtype)

        if l1_loss:
            loss = (predicted_deltas - gt_gradients).abs().mean()
        else:
            loss = ((predicted_deltas - gt_gradients) ** 2).mean()
        if clamp_large_error > 0:
            loss = loss.clamp(max=clamp_large_error)
        return loss
