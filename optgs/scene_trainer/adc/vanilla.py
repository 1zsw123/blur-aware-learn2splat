import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Bool
from torch import Tensor

from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.base import BaseStrategyCfg, GenericStrategyState
from optgs.scene_trainer.adc.base import (
    _1d_indices_from_mask,
    _densification_postfix,
    _prune_objects,
    _split_objects,
    _clone_objects,
)
from optgs.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class VanillaStrategyCfg(BaseStrategyCfg):
    """Vanilla 3DGS / EDGS densification (clone/split/prune). Adds no fields beyond the common
    base; the narrowed ``name`` lets the StrategyCfg union discriminate this arm by name."""
    name: Literal["default", "edgs", "none"]


@dataclass
class AdaptiveStrategyCfg(BaseStrategyCfg):
    """Scale-free ADC driven by the support of blur-aware screen gradients.

    The policy has no raw gradient threshold. It balances normalized residual
    demand against occupied capacity and derives its scene cap from the
    fraction that is active in the current view batch.
    """

    name: Literal["adaptive"]
    active_budget: int
    support_conditioned_cap: bool
    min_growth_fraction: float
    max_growth_fraction: float
    reward_conditioned: bool = False
    reward_ema_decay: float = 0.5


@dataclass
class VanillaStrategyState(GenericStrategyState):
    # for densification and pruning
    grad2d_norm_accum: Float[Tensor, "gaussian"]  # running accum of the norm of the image plane gradients for each GS
    denom: Float[Tensor, "gaussian"]
    radii2d: Float[Tensor, "gaussian"]  # max radius in 2D screen space observed for each GS
    scene_extent: Float[float, ""]
    initial_points: int
    last_visible_fraction: float
    last_event_step: int
    last_support_fraction: float
    last_capacity_pressure: float
    last_growth_fraction: float
    last_growth_budget: int
    last_base_cap: int
    last_demand_multiplier: float
    last_effective_cap: int
    feedback_revision: int
    feedback_probe_psnr: float
    feedback_probe_surplus: float
    feedback_has_surplus: bool
    consumed_feedback_revision: int
    previous_probe_psnr: float
    previous_probe_surplus: float
    previous_probe_has_surplus: bool
    reward_delta_count: int
    reward_psnr_delta_ema: float
    reward_psnr_abs_ema: float
    reward_surplus_delta_ema: float
    reward_surplus_abs_ema: float
    last_probe_psnr_delta: float
    last_probe_surplus_delta: float
    last_quality_reward: float
    last_complexity_cost: float
    last_densification_reward: float
    last_densification_reward_ema: float
    last_action_factor: float
    last_rewarded_support_fraction: float
    last_reward_used: bool

    def external_pruning(self, valid_points_mask: Tensor) -> None:
        if self.grad2d_norm_accum is not None:
            self.grad2d_norm_accum = self.grad2d_norm_accum[valid_points_mask]
        if self.radii2d is not None:
            self.radii2d = self.radii2d[valid_points_mask]
        if self.denom is not None:
            self.denom = self.denom[valid_points_mask]

    @classmethod
    def initialize(cls, nr_points: int, device: torch.device, scene_extent: int | float) -> "VanillaStrategyState":
        return cls(
            grad2d_norm_accum=torch.zeros(nr_points, device=device),
            denom=torch.zeros(nr_points, device=device),
            radii2d=torch.zeros(nr_points, device=device),
            scene_extent=scene_extent,
            initial_points=nr_points,
            last_visible_fraction=0.0,
            last_event_step=-1,
            last_support_fraction=0.0,
            last_capacity_pressure=0.0,
            last_growth_fraction=0.0,
            last_growth_budget=0,
            last_base_cap=nr_points,
            last_demand_multiplier=1.0,
            last_effective_cap=nr_points,
            feedback_revision=-1,
            feedback_probe_psnr=0.0,
            feedback_probe_surplus=0.0,
            feedback_has_surplus=False,
            consumed_feedback_revision=-1,
            previous_probe_psnr=0.0,
            previous_probe_surplus=0.0,
            previous_probe_has_surplus=False,
            reward_delta_count=0,
            reward_psnr_delta_ema=0.0,
            reward_psnr_abs_ema=0.0,
            reward_surplus_delta_ema=0.0,
            reward_surplus_abs_ema=0.0,
            last_probe_psnr_delta=0.0,
            last_probe_surplus_delta=0.0,
            last_quality_reward=0.0,
            last_complexity_cost=0.0,
            last_densification_reward=0.0,
            last_densification_reward_ema=0.0,
            last_action_factor=1.0,
            last_rewarded_support_fraction=0.0,
            last_reward_used=False,
        )


def update_vanilla_strategy_state(
    adc_state: VanillaStrategyState,
    radii_2d: Tensor,  # [B, V, G, 2]
    means2d_grads: Tensor | None,  # [B, V, G, 2]
    visibility_mask: Bool[Tensor, "b v gaussian"],  # [B, V, G]
    v: int,  # number of views rendered
    w: int,  # image width
    h: int,  # image height
) -> None:
    """Updates adc_state in place."""
    # get gs ids from visibility mask

    visibility_mask = visibility_mask.squeeze(0)  # [V, G], assume batch size 1
    adc_state.last_visible_fraction = float(visibility_mask.any(dim=0).float().mean())
    gs_ids = torch.where(visibility_mask)[1]  # [G_valid]
    assert visibility_mask.ndim == 2, "visibility_mask should be of shape [V, G]"

    if means2d_grads is not None:
        assert means2d_grads.ndim == 4 and means2d_grads.shape[-1] == 2, "means2d_grads should be of shape [B, V, G, 2]"
        means2d_grads = means2d_grads.squeeze(0)  # [V, G, 2], assume batch size 1
        grads = means2d_grads[visibility_mask]  # [G_valid, 2]

        # The per-renderer [-1, 1] NDC screen normalization now lives in the decoder
        # (Decoder.means2d_grad_to_ndc), so means2d_grads already arrives in NDC and this strategy
        # is renderer-agnostic. Only the view-count factor remains here.
        grads = grads * v

        # accumulate 2D grads norm
        adc_state.grad2d_norm_accum.index_add_(0, gs_ids, grads.norm(dim=-1))
        
        # accumulate denominator
        adc_state.denom.index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    radii_2d = radii_2d.squeeze(0)  # [V, N, 2], assume batch size 1
    assert radii_2d.ndim == 3 and radii_2d.shape[2] == 2, "radii_2d should be of shape [V, G, 2]"

    radii_2d = radii_2d[visibility_mask]
    radii_max = radii_2d.max(dim=-1).values  # [V, N]
    # normalize radii to [0, 1] screen space
    radii_max /= float(max(w, h))

    # update radii2d
    adc_state.radii2d[gs_ids] = torch.maximum(adc_state.radii2d[gs_ids], radii_max)


def reset_adc_state(
        adc_state: VanillaStrategyState,
) -> None:
    """Resets adc_state in place."""
    adc_state.grad2d_norm_accum.zero_()
    adc_state.denom.zero_()
    adc_state.radii2d.zero_()
    torch.cuda.empty_cache()


def set_adaptive_densification_feedback(
    adc_state: VanillaStrategyState,
    feedback: dict[str, float | int | bool],
) -> None:
    """Stage the newest fixed-probe observation for the next ADC event."""
    revision = int(feedback["revision"])
    if revision <= adc_state.feedback_revision:
        return
    probe_psnr = float(feedback["probe_psnr"])
    probe_surplus = float(feedback["probe_surplus"])
    if not math.isfinite(probe_psnr) or not math.isfinite(probe_surplus):
        raise ValueError("adaptive densification feedback must be finite")
    adc_state.feedback_revision = revision
    adc_state.feedback_probe_psnr = probe_psnr
    adc_state.feedback_probe_surplus = probe_surplus
    adc_state.feedback_has_surplus = bool(feedback["has_surplus"])


def transfer_adaptive_reward_state(
    source: VanillaStrategyState | None,
    target: VanillaStrategyState | None,
) -> None:
    """Carry delayed-reward history across optimizer phase restarts.

    Gradient/radius accumulators intentionally remain fresh. Only scalar probe
    history and the immediately preceding action cost cross the boundary.
    """
    # Other ADC strategies own independent runtime state. In particular,
    # LeGSStrategyState must be handed to transfer_legs_runtime_state instead
    # of being interpreted as this controller's global reward state.
    if not isinstance(source, VanillaStrategyState) or not isinstance(
        target, VanillaStrategyState
    ):
        return
    fields = (
        "feedback_revision",
        "feedback_probe_psnr",
        "feedback_probe_surplus",
        "feedback_has_surplus",
        "consumed_feedback_revision",
        "previous_probe_psnr",
        "previous_probe_surplus",
        "previous_probe_has_surplus",
        "reward_delta_count",
        "reward_psnr_delta_ema",
        "reward_psnr_abs_ema",
        "reward_surplus_delta_ema",
        "reward_surplus_abs_ema",
        "last_probe_psnr_delta",
        "last_probe_surplus_delta",
        "last_quality_reward",
        "last_complexity_cost",
        "last_densification_reward",
        "last_densification_reward_ema",
        "last_action_factor",
        "last_rewarded_support_fraction",
        "last_reward_used",
        "last_growth_budget",
        "last_effective_cap",
    )
    for field in fields:
        setattr(target, field, getattr(source, field))


def _normalized_delayed_reward(
    cfg: AdaptiveStrategyCfg,
    adc_state: VanillaStrategyState,
) -> float:
    """Consume one probe transition and return its bounded action multiplier.

    The previous structural action is rewarded by fixed-probe improvement over
    its complete control interval. PSNR and supported render-over-EVSSM surplus
    deltas are normalized by their within-scene absolute-delta EMAs. This keeps
    the utility scale free without a dataset-specific threshold. The fraction
    of capacity added by the action is charged as its complexity cost.
    """
    adc_state.last_reward_used = False
    adc_state.last_probe_psnr_delta = 0.0
    adc_state.last_probe_surplus_delta = 0.0
    adc_state.last_quality_reward = 0.0
    adc_state.last_complexity_cost = 0.0
    adc_state.last_densification_reward = 0.0
    adc_state.last_action_factor = 1.0
    if not cfg.reward_conditioned:
        return 1.0
    if not 0.0 <= cfg.reward_ema_decay < 1.0:
        raise ValueError("reward_ema_decay must satisfy 0 <= decay < 1")
    if adc_state.feedback_revision <= adc_state.consumed_feedback_revision:
        return 1.0

    adc_state.consumed_feedback_revision = adc_state.feedback_revision
    current_psnr = adc_state.feedback_probe_psnr
    current_surplus = adc_state.feedback_probe_surplus
    current_has_surplus = adc_state.feedback_has_surplus
    if adc_state.reward_delta_count == 0:
        adc_state.previous_probe_psnr = current_psnr
        adc_state.previous_probe_surplus = current_surplus
        adc_state.previous_probe_has_surplus = current_has_surplus
        adc_state.reward_delta_count = 1
        return 1.0

    delta_psnr = current_psnr - adc_state.previous_probe_psnr
    delta_surplus = current_surplus - adc_state.previous_probe_surplus
    use_surplus = current_has_surplus and adc_state.previous_probe_has_surplus
    adc_state.previous_probe_psnr = current_psnr
    adc_state.previous_probe_surplus = current_surplus
    adc_state.previous_probe_has_surplus = current_has_surplus

    decay = cfg.reward_ema_decay
    epsilon = 1e-8
    if adc_state.reward_delta_count == 1:
        psnr_scale = max(abs(delta_psnr), epsilon)
        surplus_scale = max(abs(delta_surplus), epsilon)
        adc_state.reward_psnr_delta_ema = delta_psnr
        adc_state.reward_psnr_abs_ema = abs(delta_psnr)
        adc_state.reward_surplus_delta_ema = delta_surplus
        adc_state.reward_surplus_abs_ema = abs(delta_surplus)
    else:
        psnr_scale = max(adc_state.reward_psnr_abs_ema, epsilon)
        surplus_scale = max(adc_state.reward_surplus_abs_ema, epsilon)
        adc_state.reward_psnr_delta_ema = (
            decay * adc_state.reward_psnr_delta_ema
            + (1.0 - decay) * delta_psnr
        )
        adc_state.reward_psnr_abs_ema = (
            decay * adc_state.reward_psnr_abs_ema
            + (1.0 - decay) * abs(delta_psnr)
        )
        adc_state.reward_surplus_delta_ema = (
            decay * adc_state.reward_surplus_delta_ema
            + (1.0 - decay) * delta_surplus
        )
        adc_state.reward_surplus_abs_ema = (
            decay * adc_state.reward_surplus_abs_ema
            + (1.0 - decay) * abs(delta_surplus)
        )

    psnr_reward = math.tanh(delta_psnr / psnr_scale)
    components = [psnr_reward]
    if use_surplus:
        components.append(math.tanh(delta_surplus / surplus_scale))
    quality_reward = sum(components) / len(components)
    complexity_cost = adc_state.last_growth_budget / max(
        1, adc_state.last_effective_cap
    )
    reward = quality_reward - complexity_cost
    if adc_state.reward_delta_count > 1:
        reward_ema = (
            decay * adc_state.last_densification_reward_ema
            + (1.0 - decay) * reward
        )
    else:
        reward_ema = reward
    # A zero reward is exactly neutral. The smooth map is bounded in (0, 2),
    # so poor actions suppress but never permanently disable future growth.
    action_factor = 2.0 / (1.0 + math.exp(-reward_ema))

    adc_state.reward_delta_count += 1
    adc_state.last_probe_psnr_delta = delta_psnr
    adc_state.last_probe_surplus_delta = delta_surplus if use_surplus else 0.0
    adc_state.last_quality_reward = quality_reward
    adc_state.last_complexity_cost = complexity_cost
    adc_state.last_densification_reward = reward
    adc_state.last_densification_reward_ema = reward_ema
    adc_state.last_action_factor = action_factor
    adc_state.last_reward_used = True
    return action_factor


def adaptive_growth_masks(
    cfg: AdaptiveStrategyCfg,
    grads: Tensor,
    scales: Tensor,
    adc_state: VanillaStrategyState,
) -> tuple[Tensor, Tensor]:
    """Select a bounded, scale-free clone/split set.

    Gradient magnitudes are converted to a probability distribution. Its
    inverse participation ratio estimates how many primitives materially
    support the current residual, so multiplying by the population produces a
    growth budget without a dataset-dependent magnitude threshold.
    """

    n = grads.numel()
    visible = (adc_state.denom > 0) & torch.isfinite(grads) & (grads > 0)
    clone_mask = torch.zeros(n, dtype=torch.bool, device=grads.device)
    split_mask = torch.zeros_like(clone_mask)

    visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
    if visible_indices.numel() == 0:
        return clone_mask, split_mask

    values = grads[visible_indices].clamp_min(torch.finfo(grads.dtype).eps)
    probabilities = values / values.sum()
    support_fraction = float(
        (1.0 / (values.numel() * probabilities.square().sum())).clamp(0.0, 1.0)
    )
    action_factor = _normalized_delayed_reward(cfg, adc_state)
    if cfg.reward_conditioned:
        rewarded_support = min(1.0, support_fraction * action_factor)
    else:
        # Preserve the pre-reward arithmetic path exactly when disabled.
        rewarded_support = support_fraction

    # The learned point transformer supplies a reference active-set budget,
    # not a universal scene cap. First undo minibatch visibility. Then solve
    # for the capacity whose fractional headroom equals the unresolved
    # residual support: (C - C_base) / C = S, hence C = C_base / (1 - S).
    # This lets a broad unresolved residual request more capacity without a
    # dataset label, while concentrated residuals stay near the learned prior.
    visible_fraction = max(adc_state.last_visible_fraction, 1.0 / max(1, n))
    base_cap = max(1, math.ceil(cfg.active_budget / visible_fraction))
    if cfg.support_conditioned_cap:
        support_headroom = max(
            1.0 - support_fraction,
            torch.finfo(grads.dtype).eps,
        )
        demand_multiplier = 1.0 / support_headroom
        if cfg.reward_conditioned:
            demand_multiplier = 1.0 + action_factor * (
                demand_multiplier - 1.0
            )
        if cfg.cap_max > 0:
            demand_multiplier = min(
                demand_multiplier,
                max(1.0, cfg.cap_max / base_cap),
            )
    else:
        demand_multiplier = 1.0
    demand_cap = math.ceil(base_cap * demand_multiplier)
    effective_cap = max(n, demand_cap)
    if cfg.cap_max > 0:
        effective_cap = min(cfg.cap_max, effective_cap)
    capacity_pressure = min(1.0, n / max(1, effective_cap))
    # Treat residual demand and occupied capacity as two dimensionless,
    # competing masses. The normalized ratio closes a large capacity deficit
    # quickly when residual support is broad, while it tends continuously to
    # zero as the active representation reaches its inferred capacity. Unlike
    # a raw gradient threshold, this law is invariant to image resolution,
    # loss scale, world units, and dataset identity.
    residual_demand = rewarded_support * (1.0 - capacity_pressure)
    growth_fraction = residual_demand / (
        residual_demand + capacity_pressure + torch.finfo(grads.dtype).eps
    )
    growth_fraction = min(
        cfg.max_growth_fraction,
        max(cfg.min_growth_fraction, growth_fraction),
    )
    available = max(0, effective_cap - n)
    desired = math.ceil(n * growth_fraction)
    budget = min(available, desired, visible_indices.numel())

    adc_state.last_event_step = -2  # caller replaces with the actual step
    adc_state.last_support_fraction = support_fraction
    adc_state.last_rewarded_support_fraction = rewarded_support
    adc_state.last_capacity_pressure = capacity_pressure
    adc_state.last_growth_fraction = growth_fraction
    adc_state.last_growth_budget = int(budget)
    adc_state.last_base_cap = int(base_cap)
    adc_state.last_demand_multiplier = float(demand_multiplier)
    adc_state.last_effective_cap = int(effective_cap)
    if budget <= 0:
        return clone_mask, split_mask

    selected_local = torch.topk(values, k=budget, sorted=False).indices
    selected = visible_indices[selected_local]
    selected_scales = scales[selected].max(dim=-1).values
    scale_partition = selected_scales.median()
    clone_selected = selected[selected_scales <= scale_partition]
    split_selected = selected[selected_scales > scale_partition]
    clone_mask[clone_selected] = True
    split_mask[split_selected] = True
    return clone_mask, split_mask

def prune(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    prune_mask: Tensor,
) -> None:
    """Gaussians are updated in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("pruning not implemented for GaussiansModule")

    # NOTE: ADC runs per-Gaussian at batch size 1, so the batch dim is squeezed off here and added back after.
    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G,

    keep_idx = _1d_indices_from_mask(~prune_mask)

    # prune gaussians
    gaussians.means = means[keep_idx].unsqueeze(0)
    gaussians.scales = scales[keep_idx].unsqueeze(0)
    gaussians.opacities = opacities[keep_idx].unsqueeze(0)
    gaussians.rotations = rotations[keep_idx].unsqueeze(0)
    gaussians.rotations_unnorm = rotations_unnorm[keep_idx].unsqueeze(0)
    gaussians.harmonics = harmonics[keep_idx].unsqueeze(0)
    if covariances is not None:
        gaussians.covariances = covariances[keep_idx].unsqueeze(0)

    # prune adc state
    adc_state.grad2d_norm_accum = adc_state.grad2d_norm_accum[keep_idx]
    adc_state.radii2d = adc_state.radii2d[keep_idx]
    adc_state.denom = adc_state.denom[keep_idx]

def splitting(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    split_mask: Tensor,
    N=2,
    revised_opacity: bool = False,
) -> None:
    """Gaussians are updated in place.
    
    revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("splitting not implemented for GaussiansModule")
    
    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    # rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 3, 3]
    rotations = F.normalize(rotations_unnorm, dim=-1)

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        opacities = torch.sigmoid(opacities)  # [G]
        scales = torch.exp(scales)  # [G, 3]
    
    sel = _1d_indices_from_mask(split_mask)
    rest = _1d_indices_from_mask(~split_mask)
    
    # get params to split
    scales = scales[sel]  # [S, 3]
    rotations = rotations[sel]  # [S, 4]
    rotations_unnorm = rotations_unnorm[sel]  # [S, 4]
    opacities = opacities[sel]  # [S]
    means = means[sel]  # [S, 3]
    harmonics = harmonics[sel]  # [S, 3, d_sh]
    if covariances is not None:
        covariances = covariances[sel]  # [S, 3, 3]
    
    def _normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
        """Convert normalized quaternion to rotation matrix.

        Args:
            quat: Normalized quaternion in wxyz convension. (..., 4)

        Returns:
            Rotation matrix (..., 3, 3)
        """
        assert quat.shape[-1] == 4, quat.shape
        w, x, y, z = torch.unbind(quat, dim=-1)
        mat = torch.stack(
            [
                1 - 2 * (y ** 2 + z ** 2),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x ** 2 + z ** 2),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x ** 2 + y ** 2),
            ],
            dim=-1,
        )
        return mat.reshape(quat.shape[:-1] + (3, 3))
    
    # new means
    rotmats = _normalized_quat_to_rotmat(rotations[:, [3, 0, 1, 2]])  # [N, 3, 3] xyzw to wxyz
    device = means.device
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(N, len(scales), 3, device=device),
    )  # [split, N, 3]
    new_means = (means + samples).reshape(-1, 3)  # [2N, 3]

    # new scales
    new_scales = (scales / 1.6).repeat(N, 1)  # [2N, 3]

    # new opacities
    if revised_opacity:
        new_opacities = (1.0 - torch.sqrt(1.0 - opacities)).repeat(N)  # [2N]
    else:
        new_opacities = opacities.repeat(N)  # [2N]

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        new_opacities = torch.logit(new_opacities)
        new_scales = torch.log(new_scales)

    # new rotations
    new_rotations = rotations.repeat(N, 1)  # [2N, 4]

    # new rotations unnorm
    new_rotations_unnorm = rotations_unnorm.repeat(N, 1)  # [2N, 4]

    # new harmonics
    new_harmonics = harmonics.repeat(N, 1, 1)  # [2N, 3, d_sh]

    # new covariances
    if covariances is not None:
        new_covariances = covariances.repeat(N, 1, 1)  # [2N, 3, 3]
    else:
        new_covariances = None

    def params_fn(v: Tensor, v_new: Tensor, dim: int) -> Tensor:
        v = v.squeeze(0)[rest].unsqueeze(0)
        return torch.cat([v, v_new.unsqueeze(0)], dim=dim)

    def state_fn(v: Tensor) -> Tensor:
        repeats = [2] + [1] * (v.dim() - 1)
        v_new = v[sel].repeat(repeats)
        return torch.cat([v[rest], v_new], dim=0)

    _densification_postfix(
        gaussians,
        adc_state,
        new_means,
        new_scales,
        new_opacities,
        new_rotations,
        new_rotations_unnorm,
        new_harmonics,
        new_covariances,
        params_fn,
        state_fn,
    )

def cloning(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    clone_mask: Tensor,
) -> None:
    """Gaussians are updated in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("cloning not implemented for GaussiansModule")

    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 3, 3]

    sel = _1d_indices_from_mask(clone_mask)

    # Clone
    new_means = means[sel]
    new_opacities = opacities[sel]
    new_scales = scales[sel]
    new_rotations_unnorm = rotations_unnorm[sel]
    new_rotations = rotations[sel]
    new_harmonics = harmonics[sel]
    new_covariances = covariances[sel] if covariances is not None else None

    def params_fn(v: Tensor, v_new: Tensor, dim: int) -> Tensor:
        return torch.cat([v, v_new.unsqueeze(0)], dim=dim)

    def state_fn(v: Tensor) -> Tensor:
        return torch.cat([v, v[sel]], dim=0)

    _densification_postfix(
        gaussians,
        adc_state,
        new_means,
        new_scales,
        new_opacities,
        new_rotations,
        new_rotations_unnorm,
        new_harmonics,
        new_covariances,
        params_fn,
        state_fn,
    )

@torch.no_grad()
def apply_vanilla_strategy(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState,
    smoothers: dict[str, Any],
    zero_t: bool = False
) -> tuple[int, int, int, float | None, float | None]:
    
    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("cloning not implemented for GaussiansModule")
    
    # Densification and Pruning
    nr_cloned, nr_splitted, nr_pruned = 0, 0, 0

    if step >= cfg.refine_stop_iter:
        return nr_cloned, nr_splitted, nr_pruned, None, None

    # Calculate average 2D grads magnitude and scales
    grads: Tensor = adc_state.grad2d_norm_accum / adc_state.denom.clamp_min(1.0)  # [G]
    max_grad2d = grads.max().item()
    max_radii = adc_state.radii2d.max().item()

    # check if should densify/prune
    if (
        step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    ):
        device = gaussians.means.device
        grow_grad2d: float = cfg.grow_grad2d
        grow_scale3d: float = cfg.grow_scale3d
        prune_scale3d: float = cfg.prune_scale3d
        prune_scale2d: float = cfg.prune_scale2d
        grow_scale2d: float = cfg.grow_scale2d
        min_opacity: float = cfg.min_opacity
        prune_zero_radii: bool = cfg.prune_zero_radii

        if cfg.do_densify:
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Densification not implemented for GaussiansModule")
                scales: Tensor = gaussians.scales  # [G, 3]
            elif isinstance(gaussians, Gaussians):
                scales: Tensor = gaussians.scales.squeeze(0)  # [G, 3]
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    scales = torch.exp(scales)  # [G, 3]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")
            if isinstance(cfg, AdaptiveStrategyCfg):
                clone_mask, split_mask = adaptive_growth_masks(
                    cfg, grads, scales, adc_state
                )
                adc_state.last_event_step = step
            else:
                # Extract points that satisfy the gradient condition
                is_grad_high: Tensor = grads > grow_grad2d  # [G]

                is_small: Tensor = scales.max(dim=-1).values <= grow_scale3d * adc_state.scene_extent

                is_large: Tensor = ~is_small

                clone_mask = is_grad_high & is_small

                split_mask = is_grad_high & is_large

                if step < cfg.refine_scale2d_stop_iter:
                    split_mask |= adc_state.radii2d > grow_scale2d

            # clone ---------------------------------------------------------------------
            
            # clone points
            cloning(
                gaussians=gaussians,
                adc_state=adc_state,
                clone_mask=clone_mask,
            )
            _clone_objects(clone_mask, smoothers, zero_t=zero_t)

            # update states
            nr_cloned = int(clone_mask.sum().item())

            # new GSs added by cloning will not be split
            split_mask = torch.cat(
                [
                    split_mask,
                    torch.zeros(nr_cloned, dtype=torch.bool, device=device),
                ]
            )

            # split ---------------------------------------------------------------------
                        
            # split points
            # No need to prune after splitting since we already removed the original points in _densification_postfix
            N = 2
            splitting(
                gaussians=gaussians,
                adc_state=adc_state,
                split_mask=split_mask,
                N=N,  # split each point into N points
            )
            _split_objects(split_mask, smoothers, N=N, zero_t=zero_t)
            nr_splitted = int(split_mask.sum().item())
            
        if cfg.do_prune:

            # prune ---------------------------------------------------------------------
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Densification not implemented for GaussiansModule")
                scales: Tensor = gaussians.scales  # [G, 3]
                opacities = gaussians.opacities  # [G]
            elif isinstance(gaussians, Gaussians):
                opacities = gaussians.opacities.squeeze(0)  # [G]
                scales = gaussians.scales.squeeze(0)  # [G, 3]
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    scales = torch.exp(scales)  # [G, 3]
                    opacities = torch.sigmoid(opacities)  # [G]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")

            # find points to prune and prune gaussians
            prune_mask = opacities < min_opacity

            if step > cfg.reset_every:
                is_too_big = scales.max(dim=-1).values > prune_scale3d * adc_state.scene_extent
                if step < cfg.refine_scale2d_stop_iter:
                    is_too_big |= adc_state.radii2d > prune_scale2d
                prune_mask = prune_mask | is_too_big

            # invisible from training views
            if prune_zero_radii:
                raise NotImplementedError("prune_zero_radii not implemented yet")

            prune(gaussians, adc_state, prune_mask)
            _prune_objects(prune_mask, objects=smoothers)

            # update states
            nr_pruned = int(prune_mask.sum().item())

        # --------------------------------------------------------------------------

        # reset adc state
        reset_adc_state(adc_state)

        if isinstance(cfg, AdaptiveStrategyCfg):
            print(
                "Adaptive Densification/Pruning "
                f"@ iter {step}: support={adc_state.last_support_fraction:.4f}, "
                f"pressure={adc_state.last_capacity_pressure:.4f}, "
                f"growth={adc_state.last_growth_fraction:.4f}, "
                f"visible={adc_state.last_visible_fraction:.4f}, "
                f"budget={adc_state.last_growth_budget}, "
                f"base_cap={adc_state.last_base_cap}, "
                f"demand_x={adc_state.last_demand_multiplier:.3f}, "
                f"reward={adc_state.last_densification_reward_ema:.3f}, "
                f"action_x={adc_state.last_action_factor:.3f}, "
                f"cap={adc_state.last_effective_cap}, cloned={nr_cloned}, "
                f"splitted={nr_splitted}, pruned={nr_pruned}, "
                f"total={gaussians.means.shape[1]}"
            )
        else:
            print(
                f"Densification/Pruning @ iter {step}: cloned {nr_cloned}, splitted {nr_splitted}, pruned {nr_pruned}, total now {gaussians.means.shape[1]}"
            )

    if cfg.do_opacity_reset:

        # Opacity reset
        if step % cfg.reset_every == 0 and step > 0:
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Opacity reset not implemented for GaussiansModule")
            elif isinstance(gaussians, Gaussians):
                opacities = gaussians.opacities
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    opacities = torch.sigmoid(opacities)  # [G]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")
            
            value = cfg.min_opacity * 2.0
            new_opacities = torch.min(opacities, torch.ones_like(opacities) * value)

            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # deactivated
                new_opacities = torch.logit(new_opacities)

            gaussians.opacities = new_opacities

            # reset momentums of opacities
            smoothers["opacities"].zero_out(zero_t=zero_t)
            print("Opacity reset @ iter", step)

    if cfg.reduce_opacity:
        # Slightly reduce opacity every few steps (from EDGS)
        if step % cfg.reduce_every == 0:
            
            opacities = gaussians.opacities
            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # activate
                opacities = torch.sigmoid(opacities)  # [G]
                
            opacities_new = opacities * cfg.reduce_factor
            
            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # deactivate
                opacities_new = torch.logit(opacities_new)
                
            gaussians.opacities = opacities_new
    
    return nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d
