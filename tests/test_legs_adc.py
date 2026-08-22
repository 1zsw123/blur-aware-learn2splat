from __future__ import annotations

import random

import torch

from optgs.experimental.api.integration.config_bridge import build_refiner_cfg
from optgs.experimental.api.integration.config_bridge import build_adam_baseline
from optgs.model.types import Gaussians
from optgs.scene_trainer.adc.legs import (
    LeGSPPOController,
    LeGSStrategyState,
    _apply_legs_final_prune,
    _blur_condition_scale,
    _compose_blur_conditioned_reward,
    _enforce_safety_cap,
    _fixed_blur_probe_views,
    _normalize_blur_quality_delta,
    _sample_official_views,
    _scatter_mean,
    _standardize_blur_features,
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


def test_blur_conditioned_legs_has_separate_eighteen_dimensional_contract() -> None:
    exact = build_refiner_cfg("legs", 800)
    conditioned = build_refiner_cfg("legs_blur", 800)

    assert exact.state_dim == 11
    assert not exact.blur_conditioned
    assert conditioned.name == "legs_blur"
    assert conditioned.state_dim == 18
    assert conditioned.blur_feature_dim == 7
    assert conditioned.blur_conditioned
    assert conditioned.refine_start_iter == exact.refine_start_iter == 500
    assert conditioned.refine_every == exact.refine_every == 100
    assert conditioned.refine_stop_iter == exact.refine_stop_iter == 15_000
    assert conditioned.cap_max == exact.cap_max == -1
    assert conditioned.blur_quality_weight == 1.0
    assert conditioned.blur_capacity_weight == 0.10
    assert conditioned.blur_condition_start_iter == 2000
    assert conditioned.blur_condition_ramp_iters == 3000


def test_blur_adapter_starts_as_an_exact_legs_residual() -> None:
    exact_cfg = build_refiner_cfg("legs", 10_000)
    blur_cfg = build_refiner_cfg("legs_blur", 10_000)
    torch.manual_seed(17)
    exact = LeGSPPOController(exact_cfg, torch.device("cpu"))
    torch.manual_seed(17)
    conditioned = LeGSPPOController(blur_cfg, torch.device("cpu"))
    local_state = torch.randn(13, 11)
    blur_state = torch.randn(13, 7)

    exact_encoded = exact._encode(local_state)
    conditioned_encoded = conditioned._encode(
        torch.cat([local_state, blur_state], dim=-1)
    )

    assert torch.equal(exact_encoded, conditioned_encoded)
    assert conditioned.blur_adapter is not None
    assert not conditioned.blur_adapter.weight.any()


def test_blur_adapter_zero_state_update_preserves_exact_legs() -> None:
    exact_cfg = build_refiner_cfg("legs", 10_000)
    blur_cfg = build_refiner_cfg("legs_blur", 10_000)
    torch.manual_seed(23)
    exact = LeGSPPOController(exact_cfg, torch.device("cpu"))
    torch.manual_seed(23)
    conditioned = LeGSPPOController(blur_cfg, torch.device("cpu"))
    local_state = torch.randn(13, 11)
    zero_blur_state = torch.zeros(13, 7)

    exact.encoder_optimizer.zero_grad(set_to_none=True)
    conditioned.encoder_optimizer.zero_grad(set_to_none=True)
    exact._encode(local_state).square().mean().backward()
    conditioned._encode(
        torch.cat([local_state, zero_blur_state], dim=-1)
    ).square().mean().backward()
    exact.encoder_optimizer.step()
    conditioned.encoder_optimizer.step()

    assert conditioned.blur_adapter is not None
    assert not conditioned.blur_adapter.weight.any()
    assert torch.equal(
        exact._encode(local_state),
        conditioned._encode(torch.cat([local_state, zero_blur_state], dim=-1)),
    )


def test_blur_condition_curriculum_is_scene_independent() -> None:
    cfg = build_refiner_cfg("legs_blur", 10_000)

    assert _blur_condition_scale(cfg, 2_000) == 0.0
    assert abs(_blur_condition_scale(cfg, 3_500) - 0.5) < 1e-8
    assert _blur_condition_scale(cfg, 5_000) == 1.0
    assert _blur_condition_scale(cfg, 50_000) == 1.0


def test_blur_feature_normalization_preserves_bounded_scene_identity() -> None:
    state = LeGSStrategyState.initialize(2, torch.device("cpu"), 1.0)
    first = torch.tensor([0.2, -0.4])
    second = torch.tensor([0.6, -0.8])

    assert torch.equal(_standardize_blur_features(state, first), first)
    normalized = _standardize_blur_features(state, second)

    assert torch.equal(normalized, second)
    assert len(state.blur_feature_history) == 2


def test_blur_reward_credits_quality_and_charges_birth_capacity() -> None:
    cfg = build_refiner_cfg("legs_blur", 2_000)
    reward = torch.zeros(5)
    actions = torch.tensor([0, 1, 2, 3, 1])
    valid = torch.tensor([True, True, True, True, False])

    (
        combined,
        fraction,
        capacity_cost,
        scale,
        support_mean,
        birth_gate_mean,
        net_direction,
    ) = _compose_blur_conditioned_reward(
        reward, actions, valid, quality_reward=1.0, cfg=cfg, step=5_000
    )

    assert fraction == 0.75
    assert capacity_cost == 0.2
    assert scale == 1.0
    assert combined[0] == 0.0
    assert abs(float(combined[1]) - 1.0 / 3.0) < 1e-6
    assert abs(float(combined[2]) - 1.0 / 3.0) < 1e-6
    assert abs(float(combined[3]) + 1.0 / 3.0) < 1e-6
    assert combined[4] == 0.0
    assert support_mean == 0.5
    assert birth_gate_mean == 1.0
    assert abs(net_direction - 1.0 / 3.0) < 1e-8


def test_blur_reward_is_inactive_during_representation_warmup() -> None:
    cfg = build_refiner_cfg("legs_blur", 10_000)
    reward = torch.tensor([0.3, -0.2, 0.1])
    actions = torch.tensor([0, 1, 2])
    valid = torch.ones(3, dtype=torch.bool)

    combined, _, _, scale, _, _, _ = _compose_blur_conditioned_reward(
        reward, actions, valid, quality_reward=-1.0, cfg=cfg, step=2_000
    )

    assert scale == 0.0
    assert torch.equal(combined, reward)


def test_blur_reward_protects_locally_supported_births_from_global_regression() -> None:
    cfg = build_refiner_cfg("legs_blur", 10_000)
    reward = torch.tensor([-2.0, 2.0])
    actions = torch.tensor([1, 1])
    valid = torch.ones(2, dtype=torch.bool)

    combined, _, _, _, support_mean, birth_gate_mean, net_direction = (
        _compose_blur_conditioned_reward(
            reward,
            actions,
            valid,
            quality_reward=-1.0,
            cfg=cfg,
            step=5_000,
        )
    )

    assert combined[1] > combined[0]
    assert (combined[1] - reward[1]) > (combined[0] - reward[0])
    assert abs(support_mean - 0.5) < 1e-6
    assert abs(birth_gate_mean - 1.0) < 1e-6
    assert net_direction == 1.0


def test_harmful_net_contraction_credits_birth_and_penalizes_prune() -> None:
    cfg = build_refiner_cfg("legs_blur", 10_000)
    reward = torch.zeros(3)
    actions = torch.tensor([1, 3, 3])
    valid = torch.ones(3, dtype=torch.bool)

    combined, _, _, _, _, _, net_direction = _compose_blur_conditioned_reward(
        reward,
        actions,
        valid,
        quality_reward=-1.0,
        cfg=cfg,
        step=5_000,
    )

    assert abs(net_direction + 1.0 / 3.0) < 1e-8
    assert combined[0] > 0.0
    assert combined[1] < 0.0
    assert combined[2] < 0.0


def test_blur_quality_delta_uses_online_rms_without_dataset_scale() -> None:
    state = LeGSStrategyState.initialize(1, torch.device("cpu"), 1.0)
    positive = _normalize_blur_quality_delta(
        state, psnr_delta=2.0, surplus_delta=0.2, has_surplus=True,
        reliability=0.8,
        device=torch.device("cpu")
    )
    negative = _normalize_blur_quality_delta(
        state, psnr_delta=-2.0, surplus_delta=-0.2, has_surplus=True,
        reliability=0.8,
        device=torch.device("cpu")
    )

    assert positive > 0.0
    assert negative < 0.0
    assert abs(positive + negative) < 1e-6


def test_blur_quality_interpolates_between_teacher_and_surplus_evidence() -> None:
    teacher_state = LeGSStrategyState.initialize(1, torch.device("cpu"), 1.0)
    surplus_state = LeGSStrategyState.initialize(1, torch.device("cpu"), 1.0)

    teacher_reliable = _normalize_blur_quality_delta(
        teacher_state,
        psnr_delta=-1.0,
        surplus_delta=1.0,
        has_surplus=True,
        reliability=0.9,
        device=torch.device("cpu"),
    )
    teacher_unreliable = _normalize_blur_quality_delta(
        surplus_state,
        psnr_delta=-1.0,
        surplus_delta=1.0,
        has_surplus=True,
        reliability=0.1,
        device=torch.device("cpu"),
    )

    assert teacher_reliable < 0.0
    assert teacher_unreliable > 0.0


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
    assert views["index"].shape == (1, 10)
    assert set(int(value) for value in views["index"].flatten()).issubset(
        set(range(100, 112))
    )
    assert len(source_indices) == len(set(source_indices)) == 10
    assert set(source_indices).issubset(set(range(100, 112)))


def test_blur_policy_uses_one_fixed_declared_probe_set() -> None:
    cfg = build_refiner_cfg("legs_blur", 10_000)
    state = LeGSStrategyState.initialize(2, torch.device("cpu"), 1.0)
    context = {
        "image": torch.arange(12).reshape(1, 12, 1, 1, 1).float(),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 12, 1, 1),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 12, 1, 1),
        "near": torch.ones(1, 12),
        "far": torch.full((1, 12), 10.0),
        "index": torch.arange(100, 112).reshape(1, 12),
        "policy_probe": torch.tensor(
            [[True, False, False, True, False, False,
              False, True, False, False, False, True]]
        ),
    }

    first, first_indices = _fixed_blur_probe_views(cfg, context, state)
    context["policy_probe"].logical_not_()
    second, second_indices = _fixed_blur_probe_views(cfg, context, state)

    assert first_indices == second_indices == [100, 103, 107, 111]
    assert torch.equal(first["index"], second["index"])


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
