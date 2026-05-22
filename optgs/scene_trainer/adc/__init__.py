from typing import Any
from optgs.model.types import Gaussians
import torch
from torch import Tensor
from jaxtyping import Bool
from optgs.scene_trainer.gaussian_module import GaussiansModule
from optgs.scene_trainer.adc.base import BaseStrategyCfg
from optgs.scene_trainer.adc.mcmc import McmcStrategyState, update_mcmc_strategy_state
from optgs.scene_trainer.adc.vanilla import VanillaStrategyState, update_vanilla_strategy_state

def init_strategy_state(
    cfg: BaseStrategyCfg,
    **kwargs
) -> VanillaStrategyState | McmcStrategyState:
    
    if cfg.name == "mcmc":
        return McmcStrategyState.initialize(
            device=kwargs["device"]
        )
    elif cfg.name in ["default", "edgs", "none"]:
        return VanillaStrategyState.initialize(
            nr_points=kwargs["nr_points"],
            device=kwargs["device"],
            scene_extent=kwargs["scene_extent"],
        )
    else:
        raise NotImplementedError(f"ADC strategy state initialization not implemented for {cfg.name}")
    
def update_strategy_state(
    adc_state: VanillaStrategyState | McmcStrategyState,
    **kwargs
) -> None:
    """Updates adc_state in place."""
    
    if isinstance(adc_state, VanillaStrategyState):
        return update_vanilla_strategy_state(
            adc_state,
            radii_2d=kwargs["radii_2d"],
            means2d_grads=kwargs["means2d_grads"],
            visibility_mask=kwargs["visibility_mask"],
            v=kwargs["v"],
            w=kwargs["w"],
            h=kwargs["h"],
        )
    elif isinstance(adc_state, McmcStrategyState):
        return update_mcmc_strategy_state(adc_state)
    else:
        raise NotImplementedError(f"ADC strategy state update not implemented for {type(adc_state)}")


def apply_adc_strategy(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState | McmcStrategyState,
    smoothers: dict[str, Any],
    zero_t: bool = False,
    **kwargs
) -> tuple[int, int, int, float | None, float | None]:
    """Applies ADC strategy and returns number of cloned, splitted, pruned GSs."""
    
    if cfg.name in ["default", "edgs", "none"]:
        from optgs.scene_trainer.adc.vanilla import apply_vanilla_strategy
        assert isinstance(adc_state, VanillaStrategyState), "adc_state type mismatch."
        return apply_vanilla_strategy(
            cfg,
            step=step,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=smoothers,
            zero_t=zero_t
        )
    elif cfg.name == "mcmc":
        from optgs.scene_trainer.adc.mcmc import apply_mcmc_strategy
        assert isinstance(adc_state, McmcStrategyState), "adc_state type mismatch."
        return apply_mcmc_strategy(
            cfg,
            step=step,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=smoothers,
            lr=kwargs["lr"],
            # zero_t=zero_t
        )
    else:
        raise NotImplementedError(f"ADC strategy not implemented for {cfg.name}")

    
@torch.no_grad()
def post_backward(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState | McmcStrategyState,
    smoothers: dict[str, Any],
    radii_2d: Tensor,  # [B, V, G, 2]
    means2d_grads: Tensor | None,  # [B, V, G, 2]
    visibility_mask: Bool[Tensor, "b v gaussian"],  # [B, V, G]
    iter_batch_size: int,
    w: int,
    h: int,
    zero_t: bool = False,
    **kwargs
) -> tuple[int, int, int, float | None, float | None]:

    # update adc state
    update_strategy_state(
        adc_state,
        radii_2d=radii_2d,
        means2d_grads=means2d_grads,
        visibility_mask=visibility_mask,
        v=iter_batch_size,
        w=w,
        h=h,
    )

    return apply_adc_strategy(
        cfg,
        step=step,
        gaussians=gaussians,
        adc_state=adc_state,
        smoothers=smoothers,
        zero_t=zero_t,
        **kwargs
    )
