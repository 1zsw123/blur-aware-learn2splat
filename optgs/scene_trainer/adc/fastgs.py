"""FastGS adaptive density control (CVPR 2026, arXiv:2511.04283).

FastGS keeps vanilla 3DGS's clone-vs-split skeleton but changes *which* Gaussians grow and die:

1. **Multi-view importance filter** (the paper's main idea): before each densification, sample a
   few training cameras, flag the high-error pixels (normalized-L1 > ``loss_thresh``), and ask the
   rasterizer which Gaussians contributed to them. Only Gaussians flagged across enough views
   (``importance_score > importance_thresh``) may densify — killing the redundant growth vanilla
   produces from a single noisy view.
2. **Abs-GS split gradient** (arXiv:2404.10484): clone uses the normal screen-grad
   (``grow_grad2d``), split uses the *absolute* screen-grad (``grad_abs_thresh``), which does not
   cancel across a Gaussian's footprint. The FastGS rasterizer emits both as a ``[N, 4]`` screen
   tensor (cols [:2] normal, cols [2:] abs).
3. **Score-guided budgeted pruning**: vanilla opacity/size eligibility, but only
   ``prune_budget_frac`` of the eligible set is removed, biased toward the worst reconstruction
   score (``pruning_score``). Opacity is clamped to ``opacity_clamp`` after each step (not reset).

This module owns the densification logic; clone/split/prune *mechanics* are reused from
``vanilla.py`` unchanged — only the masks differ. The two multi-view signals
(``importance_score``, ``pruning_score``) are computed by the caller from the FastGS decoder's
``render_metric_counts`` (the rasterizer ``get_flag``/``metric_map`` path) and reduced by
``compute_fastgs_scores``; the abs-grad arrives via the decoder's second screen-space leaf. The
test-time postprocessing loop wires all of this. The in-loop ADC dispatch (``adc/__init__.py``)
supplies neither, so there the strategy runs in its fallback: no importance filter, deterministic
pruning (``importance_score``/``pruning_score`` = None) and normal-grad splitting (no abs-grad) —
i.e. vanilla 3DGS. FastGS-specific knobs are read off ``cfg`` via ``getattr`` so
``BaseStrategyCfg`` and the other refiner configs stay untouched; for exact-FastGS parity set
``grow_scale3d=0.001``.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.base import (
    BaseStrategyCfg,
    GenericStrategyState,
    _clone_objects,
    _prune_objects,
    _split_objects,
)
from optgs.scene_trainer.adc.vanilla import cloning, prune, splitting
from optgs.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class FastGSStrategyCfg(BaseStrategyCfg):
    """FastGS densification knobs (defaults from submodules/FastGS: arguments/__init__.py +
    train.py). The narrowed ``name`` discriminates this arm of the StrategyCfg union."""
    name: Literal["fastgs"]
    grad_abs_thresh: float = 0.0012  # Abs-GS split threshold
    importance_thresh: int = 5  # min averaged high-error view count to densify (integer count)
    prune_budget_frac: float = 0.5  # fraction of eligible Gaussians actually pruned per step
    opacity_clamp: float = 0.8  # opacity ceiling applied after each densification


@dataclass
class FastGSStrategyState(GenericStrategyState):
    # Running accumulators, reset after every densification.
    grad2d_norm_accum: Float[Tensor, "gaussian"]  # normal screen-grad norm (clone signal)
    grad2d_abs_norm_accum: Float[Tensor, "gaussian"]  # abs screen-grad norm (split signal, Abs-GS)
    denom: Float[Tensor, "gaussian"]
    radii2d: Float[Tensor, "gaussian"]  # max 2D radius, normalized to [0,1] by max(w,h)
    scene_extent: Float[float, ""]
    uses_abs_grad: bool = False  # True once an abs-grad has been accumulated

    def external_pruning(self, valid_points_mask: Tensor) -> None:
        self.grad2d_norm_accum = self.grad2d_norm_accum[valid_points_mask]
        self.grad2d_abs_norm_accum = self.grad2d_abs_norm_accum[valid_points_mask]
        self.radii2d = self.radii2d[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    @classmethod
    def initialize(cls, nr_points: int, device: torch.device, scene_extent: int | float) -> "FastGSStrategyState":
        z = lambda: torch.zeros(nr_points, device=device)
        return cls(grad2d_norm_accum=z(), grad2d_abs_norm_accum=z(), denom=z(), radii2d=z(), scene_extent=scene_extent)


def update_fastgs_strategy_state(
    adc_state: FastGSStrategyState,
    radii_2d: Tensor,  # [B, V, G, 2]
    means2d_grads: Tensor | None,  # [B, V, G, 2], NDC — normal screen grad (clone signal)
    means2d_abs_grads: Tensor | None,  # [B, V, G, 2], NDC — abs screen grad (split signal)
    visibility_mask: Bool[Tensor, "b v gaussian"],  # [B, V, G]
    v: int,  # number of views rendered
    w: int,  # image width
    h: int,  # image height
) -> None:
    """Updates adc_state in place. Both gradients arrive already in NDC (the renderer-agnostic
    ``Decoder.means2d_grad_to_ndc`` normalization); only the view-count factor remains here
    (matches vanilla.py). If the abs grad is absent, splitting falls back to the normal grad."""
    visibility_mask = visibility_mask.squeeze(0)  # [V, G], assume batch size 1
    assert visibility_mask.ndim == 2, "visibility_mask should be of shape [V, G]"
    gs_ids = torch.where(visibility_mask)[1]  # [G_valid]

    def _accumulate(grads_bvg2: Tensor, accum: Tensor) -> None:
        assert grads_bvg2.ndim == 4 and grads_bvg2.shape[-1] == 2, "grads should be [B, V, G, 2]"
        grads = grads_bvg2.squeeze(0)[visibility_mask] * v  # [G_valid, 2]
        accum.index_add_(0, gs_ids, grads.norm(dim=-1))

    if means2d_grads is not None:
        _accumulate(means2d_grads, adc_state.grad2d_norm_accum)
        adc_state.denom.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

    if means2d_abs_grads is not None:
        _accumulate(means2d_abs_grads, adc_state.grad2d_abs_norm_accum)
        adc_state.uses_abs_grad = True

    radii_2d = radii_2d.squeeze(0)  # [V, G, 2], assume batch size 1
    assert radii_2d.ndim == 3 and radii_2d.shape[2] == 2, "radii_2d should be [V, G, 2]"
    radii_max = radii_2d[visibility_mask].max(dim=-1).values / float(max(w, h))  # normalized to [0,1]
    adc_state.radii2d[gs_ids] = torch.maximum(adc_state.radii2d[gs_ids], radii_max)


def reset_fastgs_state(adc_state: FastGSStrategyState) -> None:
    """Resets adc_state in place after densification. The reused vanilla clone/split/prune resize
    the other accumulators but not the FastGS abs one, so just reallocate it to the current size —
    it is zeroed here anyway, so nothing is lost."""
    adc_state.grad2d_norm_accum.zero_()
    adc_state.denom.zero_()
    adc_state.radii2d.zero_()
    adc_state.grad2d_abs_norm_accum = torch.zeros_like(adc_state.grad2d_norm_accum)
    torch.cuda.empty_cache()


def compute_fastgs_scores(
    per_view_metric_counts: list[Tensor],  # each [G]: per-Gaussian high-error pixel flag counts
    per_view_losses: list[float],  # each: scalar photometric loss of that view
    densify: bool = True,
) -> tuple[Tensor | None, Tensor]:
    """Reduce per-view rasterizer flag counts into FastGS's importance/pruning scores.

    Mirrors ``submodules/FastGS/utils/fast_utils.py::compute_gaussian_score_fastgs`` (the tensor
    reduction only). The per-view counts come from the FastGS rasterizer's ``get_flag``/
    ``metric_map`` path, which the optgs decoder must expose for this to run.

    Returns ``importance_score`` = floor(sum_v counts_v / n_views) (None if not ``densify``) and
    ``pruning_score`` = min-max normalized sum_v (loss_v * counts_v), per Gaussian in [0, 1].
    """
    n_views = len(per_view_metric_counts)
    full_counts = torch.stack(per_view_metric_counts).sum(dim=0)  # [G]
    full_score = torch.stack([l * c for l, c in zip(per_view_losses, per_view_metric_counts)]).sum(dim=0)  # [G]

    denom = (full_score.max() - full_score.min()).clamp_min(torch.finfo(full_score.dtype).eps)
    pruning_score = (full_score - full_score.min()) / denom
    importance_score = torch.div(full_counts, n_views, rounding_mode="floor") if densify else None
    return importance_score, pruning_score


@torch.no_grad()
def apply_fastgs_strategy(
    cfg: FastGSStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: FastGSStrategyState,
    smoothers: dict[str, Any],
    importance_score: Tensor | None = None,  # [G] averaged high-error view count; None = no filter
    pruning_score: Tensor | None = None,  # [G] in [0,1]; None = deterministic full prune
    zero_t: bool = False,
) -> tuple[int, int, int, float | None, float | None]:
    """FastGS densification/pruning. Returns (nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d)."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("FastGS ADC not implemented for GaussiansModule")

    nr_cloned, nr_splitted, nr_pruned = 0, 0, 0
    if step >= cfg.refine_stop_iter:
        return nr_cloned, nr_splitted, nr_pruned, None, None

    grad_abs_thresh = cfg.grad_abs_thresh
    importance_thresh = cfg.importance_thresh
    prune_budget_frac = cfg.prune_budget_frac
    opacity_clamp = cfg.opacity_clamp

    # Mean per-observation gradient norms (FastGS xyz_gradient_accum / denom).
    denom = adc_state.denom.clamp_min(1.0)
    grads: Tensor = adc_state.grad2d_norm_accum / denom  # clone signal
    if adc_state.uses_abs_grad:
        split_grads, split_thresh = adc_state.grad2d_abs_norm_accum / denom, grad_abs_thresh  # Abs-GS
    else:
        split_grads, split_thresh = grads, cfg.grow_grad2d  # no abs grad: split on the normal grad
    max_grad2d = grads.max().item()
    max_radii = adc_state.radii2d.max().item()

    device = gaussians.means.device
    densify_step = (
        step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    )

    if densify_step:
        scales: Tensor = gaussians.scales.squeeze(0)  # [G, 3]
        if not gaussians.stores_activated:
            scales = torch.exp(scales)
        is_small: Tensor = scales.max(dim=-1).values <= cfg.grow_scale3d * adc_state.scene_extent

        # Multi-view importance filter: only Gaussians consistently flagged as high-error densify.
        metric_mask = importance_score > importance_thresh if importance_score is not None \
            else torch.ones(scales.shape[0], dtype=torch.bool, device=device)

        if cfg.do_densify:
            clone_mask: Tensor = (grads > cfg.grow_grad2d) & is_small & metric_mask  # small + normal grad
            split_mask: Tensor = (split_grads > split_thresh) & ~is_small & metric_mask  # large + abs grad
            if step < cfg.refine_scale2d_stop_iter:
                split_mask |= adc_state.radii2d > cfg.grow_scale2d

            cloning(gaussians=gaussians, adc_state=adc_state, clone_mask=clone_mask)
            _clone_objects(clone_mask, smoothers, zero_t=zero_t)
            nr_cloned = int(clone_mask.sum().item())

            # New clones are never split this round.
            split_mask = torch.cat([split_mask, torch.zeros(nr_cloned, dtype=torch.bool, device=device)])
            splitting(gaussians=gaussians, adc_state=adc_state, split_mask=split_mask, N=2)
            _split_objects(split_mask, smoothers, N=2, zero_t=zero_t)
            nr_splitted = int(split_mask.sum().item())

        if cfg.do_prune:
            opacities = gaussians.opacities.squeeze(0)  # [G]
            scales = gaussians.scales.squeeze(0)  # [G, 3]
            if not gaussians.stores_activated:
                opacities = torch.sigmoid(opacities)
                scales = torch.exp(scales)

            prune_mask = opacities < cfg.min_opacity
            if step > cfg.reset_every:
                is_too_big = scales.max(dim=-1).values > cfg.prune_scale3d * adc_state.scene_extent
                if step < cfg.refine_scale2d_stop_iter:
                    is_too_big |= adc_state.radii2d > cfg.prune_scale2d
                prune_mask = prune_mask | is_too_big

            prune_mask = _budget_prune_mask(prune_mask, pruning_score, prune_budget_frac, device)
            prune(gaussians, adc_state, prune_mask)
            _prune_objects(prune_mask, objects=smoothers)
            nr_pruned = int(prune_mask.sum().item())

        # FastGS clamps opacity to a ceiling after densification (not a reset).
        _set_opacity(gaussians, lambda op: torch.minimum(op, torch.full_like(op, opacity_clamp)))

        reset_fastgs_state(adc_state)
        print(
            f"FastGS Densification/Pruning @ iter {step}: cloned {nr_cloned}, splitted {nr_splitted}, "
            f"pruned {nr_pruned}, total now {gaussians.means.shape[1]}"
        )

    # Periodic opacity reset to 2*min_opacity (FastGS reset_opacity, run alongside densification).
    if cfg.do_opacity_reset and step > 0 and step % cfg.reset_every == 0:
        _set_opacity(gaussians, lambda op: torch.minimum(op, torch.full_like(op, cfg.min_opacity * 2.0)))
        if smoothers.get("opacities") is not None:
            smoothers["opacities"].zero_out(zero_t=zero_t)
        print(f"FastGS Opacity reset @ iter {step}")

    return nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d


def _set_opacity(gaussians: Gaussians, fn) -> None:
    """Apply ``fn`` to the *activated* opacities and write them back in the stored space."""
    activated = gaussians.stores_activated
    op = gaussians.opacities if activated else torch.sigmoid(gaussians.opacities)
    op = fn(op)
    gaussians.opacities = op if activated else torch.logit(op)


def _budget_prune_mask(
    prune_mask: Tensor,  # [G_now] bool, eligibility (post-densify count)
    pruning_score: Tensor | None,  # [G_pre] in [0,1], measured pre-densify; None = prune all eligible
    budget_frac: float,
    device: torch.device,
) -> Tensor:
    """Keep only ``budget_frac`` of the eligible Gaussians, sampled with bias toward the worst
    reconstruction score (FastGS densify_and_prune). ``pruning_score`` is measured on the
    pre-densify Gaussians, so — as in FastGS — pad/truncate it to the current count (new Gaussians
    get a neutral score and are never preferentially pruned)."""
    n_eligible = int(prune_mask.sum().item())
    if pruning_score is None or n_eligible == 0:
        return prune_mask

    remove_budget = int(budget_frac * n_eligible)
    if remove_budget == 0:
        return torch.zeros_like(prune_mask)

    score = torch.ones(prune_mask.shape[0], device=device)  # neutral score for new/unscored Gaussians
    m = min(pruning_score.shape[0], score.shape[0])
    score[:m] = pruning_score[:m]
    weights = prune_mask.float() / (1e-6 + (1.0 - score).clamp_min(0.0))  # non-eligible -> weight 0
    sampled = torch.multinomial(weights, min(remove_budget, n_eligible), replacement=False)

    budget_mask = torch.zeros_like(prune_mask)
    budget_mask[sampled] = True
    return prune_mask & budget_mask
