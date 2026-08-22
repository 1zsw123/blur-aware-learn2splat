from __future__ import annotations

import random

import torch

from optgs.experimental.api.integration.config_bridge import build_refiner_cfg
from optgs.experimental.api.integration.config_bridge import build_adam_baseline
from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.legs import (
    LeGSStrategyState,
    _apply_legs_final_prune,
    _enforce_safety_cap,
    _sample_official_views,
    _scatter_mean,
    transfer_legs_runtime_state,
)
from optgs.scene_trainer.adc.vanilla import transfer_adaptive_reward_state


def _gaussians(opacities: list[float]) -> Gaussians:
    count = len(opacities)
    rotations = torch.zeros(1, count, 4)
    rotations[..., 3] = 1.0
    return Gaussians(
        means=torch.zeros(1, count, 3),
        covariances=None,
        harmonics=torch.zeros(1, count, 3, 1),
        opacities=torch.tensor([opacities]),
        scales=torch.ones(1, count, 3),
        rotations=rotations,
        rotations_unnorm=rotations.clone(),
        stores_activated=True,
    )


def test_exact_legs_schedule_is_not_rescaled_by_smoke_horizon() -> None:
    short = build_refiner_cfg("legs", 800)
    full = build_refiner_cfg("legs", 50_000)

    for cfg in (short, full):
        assert cfg.refine_start_iter == 500
        assert cfg.refine_every == 100
        assert cfg.refine_stop_iter == 15_000
        assert cfg.reset_every == 3_000
        assert cfg.cap_max == -1
        assert cfg.state_view_count == 10
        assert cfg.ppo_chunk_size == 500_000


def test_exact_legs_is_valid_for_adam_projection_phase() -> None:
    optimizer = build_adam_baseline(10_000, adc="legs")
    assert optimizer.cfg.refiner.name == "legs"
    assert optimizer.cfg.refiner.refine_every == 100


def test_legs_runtime_transfer_is_not_routed_through_adaptive_state() -> None:
    source = LeGSStrategyState.initialize(4, torch.device("cpu"), 1.0)
    target = LeGSStrategyState.initialize(4, torch.device("cpu"), 1.0)
    source.last_event_step = 700
    source.last_clone_count = 3

    # The generic API calls both guarded transfer helpers at an optimizer
    # phase boundary. Only the LeGS helper may consume this state.
    transfer_adaptive_reward_state(source, target)
    transfer_legs_runtime_state(source, target)

    assert target.last_event_step == 700
    assert target.last_clone_count == 3


def test_exact_legs_samples_ten_training_views() -> None:
    cfg = build_refiner_cfg("legs", 10_000)
    context = {
        "image": torch.arange(12).reshape(1, 12, 1, 1, 1).float(),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 12, 1, 1),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 12, 1, 1),
        "near": torch.ones(1, 12),
        "far": torch.full((1, 12), 10.0),
        "index": torch.arange(100, 112).reshape(1, 12),
    }
    random.seed(7)
    views, source_indices = _sample_official_views(cfg, context)

    assert views["image"].shape[1] == 10
    assert len(source_indices) == len(set(source_indices)) == 10
    assert set(source_indices).issubset(set(range(100, 112)))


def test_parent_child_scatter_mean_matches_legs_credit_assignment() -> None:
    children = torch.tensor([[2.0], [4.0], [8.0], [10.0]])
    parent = torch.tensor([0, 0, 1, 1])
    result = _scatter_mean(children, parent, dim_size=2)
    assert torch.equal(result, torch.tensor([[3.0], [9.0]]))


def test_no_cap_keeps_all_legs_birth_actions() -> None:
    cfg = build_refiner_cfg("legs", 10_000)
    actions = torch.tensor([0, 1, 2, 1, 2])
    confidence = torch.linspace(0.1, 0.9, actions.numel())
    original = actions.clone()

    assert _enforce_safety_cap(cfg, actions, confidence) == 0
    assert torch.equal(actions, original)


def test_exact_legs_final_prune_uses_point_one_opacity() -> None:
    cfg = build_refiner_cfg("legs", 50_000)
    gaussians = _gaussians([0.04, 0.10, 0.3])
    state = LeGSStrategyState.initialize(3, torch.device("cpu"), 1.0)
    state.grad2d_norm_accum.fill_(1.0)
    state.grad2d_abs_norm_accum.fill_(2.0)
    state.denom.fill_(3.0)
    state.radii2d.fill_(4.0)
    state.parent_mapping = torch.arange(3)

    pruned = _apply_legs_final_prune(
        cfg, 18_000, gaussians, state, smoothers={}
    )

    assert pruned == 1
    assert gaussians.means.shape[1] == 2
    assert state.grad2d_norm_accum.shape == (2,)
    assert state.grad2d_abs_norm_accum.shape == (2,)
    assert state.denom.shape == (2,)
    assert state.radii2d.shape == (2,)
    assert state.parent_mapping.shape == (2,)
    assert not state.grad2d_norm_accum.any()
    assert not state.grad2d_abs_norm_accum.any()
    assert state.last_event_kind == "final_prune"
