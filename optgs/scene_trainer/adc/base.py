from dataclasses import dataclass
from typing import Literal, Callable
from torch import Tensor
from optgs.scene_trainer.gaussian_module import GaussiansModule
from optgs.model.types import Gaussians

@dataclass
class GenericStrategyState:
    pass

@dataclass
class BaseStrategyCfg:
    name: Literal["default", "edgs", "mcmc", "none"]
    
    do_densify: bool
    do_prune: bool
    do_opacity_reset: bool
    
    cap_max: int  # Maximum number of GSs, -1 for no cap
    noise_lr: float  # MCMC samping noise learning rate, 0.0 for no MCMC sampling
    
    pause_refine_after_reset: int
    refine_every: int
    reset_every: int
    refine_start_iter: int
    refine_stop_iter: int
    refine_scale2d_stop_iter: int  # Until which iteration 2D scale based refinement / pruning is applied
    
    grow_grad2d: float # GSs with image plane gradient above this value will be split/duplicated
    grow_scale3d: float # GSs with scale below this value will be duplicated. Above will be split
    prune_scale3d: float # GSs with scale above this value will be pruned
    prune_scale2d: float  # GSs with 2d scale (normalized by image resolution) above this value will be pruned
    grow_scale2d: float  # GSs with 2d scale (normalized by image resolution) above this value will be split
    min_opacity: float  # GSs with opacity below this value will be pruned
    prune_zero_radii: bool  # GSs with zero radii in screen space will be pruned

    reduce_opacity: bool # Slightly reduce opacity every few steps
    reduce_factor: float # Factor to reduce opacity by
    reduce_every: int # Reduce opacity every N iterations

    # Fallback means lr used for MCMC noise injection when the optimizer has no means_lr_scheduler.
    # Matches the original paper's intended scale: means_lr (~1.6e-4) * noise_lr (5e5) ≈ 80 world units.
    fallback_means_lr: float

    # If True, relocated Gaussians inherit the optimizer state of the alive Gaussian they were
    # sampled from (better initialization). If False, state is zeroed (original paper behaviour).
    relocate_copy_state: bool = False

    # MCMC noise: cap on the scales used for the noise covariance (does NOT affect rendered scales).
    # Needed because knn_based saturates clamp_max_scale, producing covariances orders of
    # magnitude larger than vanilla's Adam-evolved scales. The resulting MCMC noise overflows the
    # renderer's tile-binning math and causes silent CUDA OOB. Rule of thumb: ~scene_scale / 5.
    noise_scale_cap: float = 1.0


def _prune_objects(prune_mask, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].prune(prune_mask)

def _clone_objects(clone_mask, objects, zero_t):
    for key in objects:
        if objects[key] is not None:
            objects[key].clone(clone_mask, zero_t)

def _split_objects(split_mask, objects, N, zero_t):
    for key in objects:
        if objects[key] is not None:
            objects[key].split(split_mask, N, zero_t)

def _add_to_objects(nr_new, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].add(nr_new)

def _replace_objects(dest_indices, from_indices, objects, zero_t):
    for key in objects:
        if objects[key] is not None:
            objects[key].replace(from_indices, dest_indices, zero_t)

def _1d_indices_from_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    return mask.nonzero(as_tuple=True)[0]
            
            
def _densification_postfix(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,
    new_means: Tensor,
    new_scales: Tensor,
    new_opacities: Tensor,
    new_rotations: Tensor,
    new_rotations_unnorm: Tensor,
    new_harmonics: Tensor,
    new_covariances: Tensor | None,
    params_fn: Callable[[Tensor], Tensor],
    state_fn: Callable[[Tensor], Tensor],
) -> None:
    """Updates gaussians and adc_state in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("_densification_postfix not implemented for GaussiansModule")

    # update gaussians
    gaussians.means = params_fn(gaussians.means, new_means, dim=1)
    gaussians.scales = params_fn(gaussians.scales, new_scales, dim=1)
    gaussians.opacities = params_fn(gaussians.opacities, new_opacities, dim=1)
    gaussians.rotations = params_fn(gaussians.rotations, new_rotations, dim=1)
    gaussians.rotations_unnorm = params_fn(gaussians.rotations_unnorm, new_rotations_unnorm, dim=1)
    gaussians.harmonics = params_fn(gaussians.harmonics, new_harmonics, dim=1)
    if gaussians.covariances is not None and new_covariances is not None:
        gaussians.covariances = params_fn(gaussians.covariances, new_covariances, dim=1)

    # update adc state
    adc_state.grad2d_norm_accum = state_fn(adc_state.grad2d_norm_accum)
    adc_state.denom = state_fn(adc_state.denom)
    adc_state.radii2d = state_fn(adc_state.radii2d)
