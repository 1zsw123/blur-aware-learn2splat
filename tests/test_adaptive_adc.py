from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.blur_aware_cross_dataset.run_cross_dataset import (
    CameraPreprocessor,
    blur_kernel_statistics,
    build_valid_mask,
    depth_measurement_to_z,
    resize_depth_preserve_samples,
    resolve_evaluation_indices,
    resolve_eval_steps,
    resolve_depth_samples_per_view,
    resolve_initialization_budget,
    resolve_auxiliary_supervision,
    resolve_sharp_supervision,
    resolve_update_budget,
    stratified_kernel_indices,
)
from experiments.blur_aware_cross_dataset.protocols import (
    resolve_tum_i2slam_protocol,
)
from optgs.dataset.data_types import BatchedViews
from optgs.experimental.api import OptGS
from optgs.experimental.blur_aware import BlurAwareObjective, BlurAwareObjectiveConfig
from optgs.scene_trainer.adc.vanilla import (
    AdaptiveStrategyCfg,
    VanillaStrategyState,
    adaptive_growth_masks,
    set_adaptive_densification_feedback,
    transfer_adaptive_reward_state,
)


class _BudgetFixture:
    reference_initial_gaussians = 70_000

    class _Optimizer:
        class _Cfg:
            max_active_gaussians = 100_000

        cfg = _Cfg()

    optimizer = _Optimizer()


def test_initialization_budget_comes_from_checkpoint_by_default() -> None:
    budget, source = resolve_initialization_budget(0, _BudgetFixture())
    assert budget == 70_000
    assert source == "checkpoint_reference_initialization"


def test_initialization_budget_keeps_explicit_rollback() -> None:
    budget, source = resolve_initialization_budget(70_000, _BudgetFixture())
    assert budget == 70_000
    assert source == "explicit_override"


def test_depth_quota_ignores_rgb_only_auxiliary_views() -> None:
    needed = 19_835
    assert resolve_depth_samples_per_view(needed, 42) == 473
    assert resolve_depth_samples_per_view(needed, 0) == 0


def test_update_budget_preserves_checkpoint_view_exposure() -> None:
    small, small_source = resolve_update_budget(0, 2000, 28, 64)
    large, large_source = resolve_update_budget(0, 2000, 3397, 64)

    assert small == 2000
    assert large == 106157
    assert small_source == large_source == "checkpoint_view_exposure"
    assert resolve_update_budget(500, 2000, 3397, 64) == (500, "explicit_override")


def test_update_budget_uses_supervision_risk_not_dataset_identity() -> None:
    mass = torch.cat((torch.full((125,), 10.0), torch.ones(3272)))
    steps, source = resolve_update_budget(0, 2000, len(mass), 64, mass)

    assert steps == 14_132
    assert source == "checkpoint_supervision_risk_exposure"


def test_eval_steps_accept_absolute_and_relative_schedule() -> None:
    assert resolve_eval_steps("25%,1000,75%", 2000) == [500, 1000, 1500, 2000]


def test_sharp_w10_is_relative_and_batch_scale_invariant() -> None:
    objective = BlurAwareObjective(
        3, BlurAwareObjectiveConfig(sharp_supervision_weight=10.0)
    )
    weights = objective._supervision_weights(
        torch.tensor([[True, False, False], [True, True, True]])
    )

    assert torch.allclose(weights.mean(dim=1), torch.ones(2))
    assert torch.allclose(weights[0, 0] / weights[0, 1], torch.tensor(10.0))
    assert torch.allclose(weights[1], torch.ones(3))


def test_sharp_w10_is_not_applied_twice_when_sampler_realizes_it() -> None:
    objective = BlurAwareObjective(
        3,
        BlurAwareObjectiveConfig(
            sharp_supervision_weight=10.0,
            sharp_weight_in_sampler=True,
        ),
    )
    weights = objective._supervision_weights(
        torch.tensor([[True, False, False]])
    )
    assert torch.equal(weights, torch.ones_like(weights))


def test_coupled_dual_bpn_shares_mode_and_orders_blur_strength() -> None:
    objective = BlurAwareObjective(
        3, BlurAwareObjectiveConfig(coupled_dual_bpn=True)
    )
    family = objective.bpn.kernel_family(torch.tensor([[0, 1, 2]]))

    assert torch.allclose(
        family["teacher_kernels"].sum(dim=-1), torch.ones(3)
    )
    assert torch.allclose(family["raw_kernels"].sum(dim=-1), torch.ones(3))
    assert torch.all(family["raw_strength"] >= family["teacher_strength"])
    identity = torch.zeros_like(family["base_kernels"])
    identity[:, identity.shape[-1] // 2] = 1.0
    teacher_direction = family["teacher_kernels"] - identity
    raw_direction = family["raw_kernels"] - identity
    # Both branches can only move along the same learned blur mode.
    assert torch.allclose(
        teacher_direction * family["raw_strength"][:, None],
        raw_direction * family["teacher_strength"][:, None],
        atol=1e-7,
    )


def test_single_bpn_rollback_keeps_identity_teacher_and_original_raw_kernel() -> None:
    objective = BlurAwareObjective(
        2, BlurAwareObjectiveConfig(coupled_dual_bpn=False)
    )
    indices = torch.tensor([[0, 1]])
    family = objective.bpn.kernel_family(indices)
    embedding = objective.bpn.camera_embedding(indices.reshape(-1))
    original = torch.softmax(objective.bpn.kernel_head(embedding), dim=-1)
    identity = torch.zeros_like(original)
    identity[:, identity.shape[-1] // 2] = 1.0

    assert torch.equal(family["teacher_kernels"], identity)
    assert torch.equal(family["raw_kernels"], original)


def _checkerboard(size: int = 32) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    image = ((x + y) % 2).float().view(1, 1, 1, size, size)
    return image.repeat(1, 1, 3, 1, 1)


def _vertical_stripes(size: int = 32, stripe_width: int = 4) -> torch.Tensor:
    x = torch.arange(size)
    image = ((x // stripe_width) % 2).float().view(1, 1, 1, 1, size)
    image = image.expand(1, 1, 1, size, size)
    return image.repeat(1, 1, 3, 1, 1)


def test_laplacian_energy_loss_penalizes_render_below_evssm_gain() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.25 * (_checkerboard() - 0.5)

    below, stats = BlurAwareObjective._laplacian_energy_objective(
        raw.clone(), raw, target, torch.tensor([[False]]), None
    )
    matched, _ = BlurAwareObjective._laplacian_energy_objective(
        target.clone(), raw, target, torch.tensor([[False]]), None
    )

    assert float(below) > 0.0
    assert torch.equal(matched, torch.zeros_like(matched))
    assert float(stats["laplacian_relative_gain"]) < 0.0


def test_laplacian_energy_loss_does_not_penalize_valid_evssm_surplus() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.1 * (_checkerboard() - 0.5)
    sharper = 0.5 + 0.3 * (_checkerboard() - 0.5)

    loss, stats = BlurAwareObjective._laplacian_energy_objective(
        sharper, raw, target, torch.tensor([[False]]), None
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    assert float(stats["laplacian_relative_gain"]) > 0.0


def test_laplacian_energy_loss_matches_authoritative_sharp_raw() -> None:
    raw = 0.5 + 0.2 * (_checkerboard() - 0.5)
    same, _ = BlurAwareObjective._laplacian_energy_objective(
        raw.clone(), raw, raw, torch.tensor([[True]]), None
    )
    blurred = torch.full_like(raw, 0.5)
    degraded, _ = BlurAwareObjective._laplacian_energy_objective(
        blurred, raw, raw, torch.tensor([[True]]), None
    )

    assert torch.equal(same, torch.zeros_like(same))
    assert float(degraded) > 0.0


def test_spatial_laplacian_rejects_equal_energy_at_wrong_location() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.25 * (_vertical_stripes() - 0.5)
    shifted = torch.roll(target, shifts=4, dims=-1)

    matched, _ = BlurAwareObjective._laplacian_spatial_objective(
        target,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
    )
    misplaced, _ = BlurAwareObjective._laplacian_spatial_objective(
        shifted,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
    )

    assert torch.equal(matched, torch.zeros_like(matched))
    assert float(misplaced) > 0.0


def test_spatial_laplacian_confidence_continuously_gates_evssm() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.25 * (_vertical_stripes() - 0.5)
    prediction = raw.clone()

    full, _ = BlurAwareObjective._laplacian_spatial_objective(
        prediction,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
    )
    quarter, _ = BlurAwareObjective._laplacian_spatial_objective(
        prediction,
        raw,
        target,
        torch.tensor([[False]]),
        torch.full((1, 1), 0.25),
        None,
    )

    assert torch.allclose(quarter, 0.25 * full)


def test_spatial_laplacian_does_not_duplicate_known_sharp_supervision() -> None:
    raw = 0.5 + 0.25 * (_vertical_stripes() - 0.5)
    blurred = torch.full_like(raw, 0.5)

    loss, _ = BlurAwareObjective._laplacian_spatial_objective(
        blurred,
        raw,
        raw,
        torch.tensor([[True]]),
        torch.ones(1, 1),
        None,
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_auxiliary_direct_supervision_scales_reconstruction_confidence() -> None:
    objective = BlurAwareObjective(
        1,
        BlurAwareObjectiveConfig(laplacian_loss_weight=0.0),
    )
    target = torch.ones(1, 1, 3, 4, 4)
    prediction = torch.zeros_like(target)
    context = {
        "image": target,
        "raw_image": target,
        "target_confidence": torch.ones(1, 1),
        "known_sharp": torch.zeros(1, 1, dtype=torch.bool),
        "direct_supervision": torch.ones(1, 1, dtype=torch.bool),
        "supervision_confidence": torch.full((1, 1), 0.2),
        "index": torch.zeros(1, 1, dtype=torch.long),
    }
    loss = objective.compute_loss(
        context=context,
        output_renderer=SimpleNamespace(color=prediction, depth=torch.ones(1, 1, 4, 4)),
        start=0,
        stop=1,
        reduction="mean",
        with_ssim=False,
        fallback_loss=lambda *args, **kwargs: torch.tensor(float("nan")),
    )

    assert torch.allclose(loss, torch.tensor(0.2))
    assert objective.last_diagnostics["bpn_bypassed"] == 1.0
    assert objective.last_diagnostics["sharp_fraction"] == 0.0


def test_auxiliary_supervision_manifest_is_independent_of_sharp_w10(tmp_path) -> None:
    manifest = tmp_path / "auxiliary.json"
    manifest.write_text(
        '{"views":[{"image_name":"virtual.png","confidence":0.2}]}'
    )

    confidence, sources = resolve_auxiliary_supervision(
        {"auxiliary_supervision_manifest": str(manifest)}
    )

    assert confidence == {"virtual": 0.2}
    assert sources == ["manifest:auxiliary.json"]


def test_spatial_laplacian_short_circuits_an_all_sharp_batch() -> None:
    objective = BlurAwareObjective(
        1,
        BlurAwareObjectiveConfig(
            laplacian_loss_weight=0.1,
            laplacian_loss_mode="spatial",
        ),
    )
    # A 1x1 image cannot pass through the reflect-padded pyramid. Successful
    # evaluation therefore also verifies that the expensive path was skipped.
    image = torch.full((1, 1, 3, 1, 1), 0.5)
    loss, stats = objective._laplacian_objective(
        image,
        image,
        image,
        torch.tensor([[True]]),
        torch.ones(1, 1),
        None,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(
        stats["dynamic_uncertainty"],
        torch.zeros_like(stats["dynamic_uncertainty"]),
    )
    assert torch.equal(stats["effective_confidence"], torch.ones(1, 1))


def test_spatial_laplacian_has_finite_aligned_gradient() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.25 * (_vertical_stripes() - 0.5)
    prediction = raw.clone().requires_grad_(True)

    loss, _ = BlurAwareObjective._laplacian_spatial_objective(
        prediction,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
    )
    loss.sum().backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0.0


def test_surplus_laplacian_uses_evssm_as_a_one_sided_edge_floor() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.2 * (_vertical_stripes() - 0.5)
    stronger = 0.5 + 0.3 * (_vertical_stripes() - 0.5)
    misplaced = torch.roll(stronger, shifts=2, dims=-1)
    cfg = BlurAwareObjectiveConfig(
        laplacian_loss_mode="surplus",
        surplus_ema_decay=0.5,
    )

    aligned_objective = BlurAwareObjective(1, cfg, torch.tensor([False]))
    aligned, aligned_stats = aligned_objective._laplacian_surplus_objective(
        stronger,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
        torch.tensor([[0]]),
    )
    misplaced_objective = BlurAwareObjective(1, cfg, torch.tensor([False]))
    wrong, wrong_stats = misplaced_objective._laplacian_surplus_objective(
        misplaced,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
        torch.tensor([[0]]),
    )

    assert torch.equal(
        aligned_stats["laplacian_floor"],
        torch.zeros_like(aligned_stats["laplacian_floor"]),
    )
    assert float(aligned_stats["laplacian_relative_gain"]) > 0.0
    assert float(wrong_stats["laplacian_floor"]) > 0.0
    assert float(wrong) > float(aligned)


def test_surplus_laplacian_softly_anchors_reliable_teacher_overshoot() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.2 * (_vertical_stripes() - 0.5)
    stronger = 0.5 + 0.3 * (_vertical_stripes() - 0.5)
    cfg = BlurAwareObjectiveConfig(
        laplacian_loss_mode="surplus",
        surplus_ema_decay=0.5,
    )

    reliable = BlurAwareObjective(1, cfg, torch.tensor([False]))
    reliable_loss, reliable_stats = reliable._laplacian_surplus_objective(
        stronger,
        raw,
        target,
        torch.tensor([[False]]),
        torch.full((1, 1), 0.9),
        None,
        torch.tensor([[0]]),
    )
    uncertain = BlurAwareObjective(1, cfg, torch.tensor([False]))
    uncertain_loss, uncertain_stats = uncertain._laplacian_surplus_objective(
        stronger,
        raw,
        target,
        torch.tensor([[False]]),
        torch.full((1, 1), 0.2),
        None,
        torch.tensor([[0]]),
    )

    assert float(reliable_stats["laplacian_overshoot"]) > 0.0
    assert float(uncertain_stats["surplus_target"]) > float(
        reliable_stats["surplus_target"]
    )
    assert float(reliable_loss) > float(uncertain_loss)


def test_dynamic_evssm_uncertainty_requires_stable_cross_view_surplus() -> None:
    cfg = BlurAwareObjectiveConfig(
        laplacian_loss_mode="surplus",
        surplus_ema_decay=0.5,
    )
    static = torch.full((1, 3), 0.8)
    indices = torch.arange(3).view(1, 3)
    non_sharp = torch.zeros(1, 3, dtype=torch.bool)

    positive = BlurAwareObjective(3, cfg, torch.zeros(3, dtype=torch.bool))
    for _ in range(8):
        effective, positive_stats = positive._update_surplus_reliability(
            torch.ones(1, 3), static, non_sharp, indices
        )

    negative = BlurAwareObjective(3, cfg, torch.zeros(3, dtype=torch.bool))
    for _ in range(8):
        unchanged, negative_stats = negative._update_surplus_reliability(
            -torch.ones(1, 3), static, non_sharp, indices
        )

    assert torch.all(effective < static)
    assert float(positive_stats["dynamic_uncertainty"].mean()) > 0.0
    assert float(positive_stats["surplus_target"].mean()) > 0.0
    assert torch.equal(unchanged, static)
    assert torch.equal(
        negative_stats["dynamic_uncertainty"],
        torch.zeros_like(negative_stats["dynamic_uncertainty"]),
    )


def test_surplus_laplacian_short_circuits_an_all_sharp_batch() -> None:
    objective = BlurAwareObjective(
        1,
        BlurAwareObjectiveConfig(
            laplacian_loss_weight=0.1,
            laplacian_loss_mode="surplus",
        ),
        torch.tensor([True]),
    )
    image = torch.full((1, 1, 3, 1, 1), 0.5)
    loss, stats = objective._laplacian_objective(
        image,
        image,
        image,
        torch.tensor([[True]]),
        torch.ones(1, 1),
        None,
        torch.tensor([[0]]),
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(
        stats["dynamic_uncertainty"],
        torch.zeros_like(stats["dynamic_uncertainty"]),
    )


def test_fixed_probe_surplus_detects_supported_render_gain() -> None:
    raw = torch.full((1, 1, 3, 32, 32), 0.5)
    target = 0.5 + 0.2 * (_vertical_stripes() - 0.5)
    stronger = 0.5 + 0.3 * (_vertical_stripes() - 0.5)

    measured = BlurAwareObjective.measure_probe_surplus(
        stronger,
        raw,
        target,
        torch.tensor([[False]]),
        torch.ones(1, 1),
        None,
    )
    sharp = BlurAwareObjective.measure_probe_surplus(
        stronger,
        raw,
        target,
        torch.tensor([[True]]),
        torch.ones(1, 1),
        None,
    )

    assert measured["has_surplus"] is True
    assert float(measured["surplus"]) > 0.0
    assert sharp["has_surplus"] is False
    assert float(sharp["surplus"]) == 0.0


def test_supervision_fps_realizes_scene_level_w10_without_duplicates() -> None:
    num_views, batch_size = 20, 4
    known_sharp = torch.zeros(1, num_views, dtype=torch.bool)
    known_sharp[:, :4] = True
    sampling_mass = torch.where(
        known_sharp,
        torch.full((1, num_views), 10.0),
        torch.ones(1, num_views),
    )
    c2w = torch.eye(4).repeat(1, num_views, 1, 1)
    c2w[0, :, 0, 3] = torch.arange(num_views)
    views = BatchedViews.from_dict(
        {
            "extrinsics": c2w,
            "intrinsics": torch.eye(3).repeat(1, num_views, 1, 1),
            "image": torch.zeros(1, num_views, 3, 2, 2),
            "near": torch.full((1, num_views), 0.01),
            "far": torch.full((1, num_views), 100.0),
            "index": torch.arange(num_views).unsqueeze(0),
            "known_sharp": known_sharp,
            "sampling_mass": sampling_mass,
        }
    )
    optgs = object.__new__(OptGS)
    optgs._reset_supervision_sampler()

    sharp_draws = 0
    total_draws = 0
    for _ in range(100):
        batch = optgs._supervision_fps_minibatch(views, batch_size)
        indices = batch.index[0]
        assert indices.unique().numel() == batch_size
        sharp_draws += int(batch.known_sharp.sum())
        total_draws += batch_size

    expected = 40.0 / (40.0 + 16.0)
    assert abs(sharp_draws / total_draws - expected) <= 1.0 / total_draws


def test_supervision_fps_repeats_rare_sharp_view_to_realize_w10() -> None:
    num_views, batch_size = 42, 8
    known_sharp = torch.zeros(1, num_views, dtype=torch.bool)
    known_sharp[:, 10] = True
    sampling_mass = torch.where(
        known_sharp,
        torch.full((1, num_views), 10.0),
        torch.ones(1, num_views),
    )
    c2w = torch.eye(4).repeat(1, num_views, 1, 1)
    c2w[0, :, 0, 3] = torch.arange(num_views)
    views = BatchedViews.from_dict(
        {
            "extrinsics": c2w,
            "intrinsics": torch.eye(3).repeat(1, num_views, 1, 1),
            "image": torch.zeros(1, num_views, 3, 2, 2),
            "near": torch.full((1, num_views), 0.01),
            "far": torch.full((1, num_views), 100.0),
            "index": torch.arange(num_views).unsqueeze(0),
            "known_sharp": known_sharp,
            "sampling_mass": sampling_mass,
        }
    )
    optgs = object.__new__(OptGS)
    optgs._reset_supervision_sampler()

    sharp_draws = 0
    total_draws = 0
    saw_repeat = False
    for _ in range(100):
        batch = optgs._supervision_fps_minibatch(views, batch_size)
        indices = batch.index[0]
        saw_repeat |= indices.unique().numel() < batch_size
        sharp_draws += int(batch.known_sharp.sum())
        total_draws += batch_size

    expected = 10.0 / (10.0 + 41.0)
    assert saw_repeat
    assert abs(sharp_draws / total_draws - expected) <= 1.0 / total_draws


def test_camera_preprocess_preserves_calibration_geometry() -> None:
    preprocessor = CameraPreprocessor(
        source_size=(640, 480),
        pre_crop=(0, 0, 0, 0),
        resize_size=(528, 400),
        crop=(8, 8, 8, 8),
        distortion=(),
        undistort_depth=True,
    )
    intrinsics = torch.tensor(
        [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]
    )
    transformed = preprocessor.transform_intrinsics(intrinsics)
    assert preprocessor.output_size == (512, 384)
    assert torch.allclose(
        transformed,
        torch.tensor(
            [
                [429.7425, 0.0, 260.2075],
                [0.0, 434.1667, 200.0833],
                [0.0, 0.0, 1.0],
            ]
        ),
        atol=1e-4,
    )


def test_identity_camera_preprocess_keeps_complete_valid_domain() -> None:
    preprocessor = CameraPreprocessor(
        source_size=(8, 6),
        pre_crop=(0, 0, 0, 0),
        resize_size=(8, 6),
        crop=(0, 0, 0, 0),
        distortion=(),
        undistort_depth=True,
    )
    intrinsics = torch.tensor(
        [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]]
    )
    valid = build_valid_mask(intrinsics, preprocessor)
    assert valid.shape == (1, 6, 8)
    assert bool(valid.all())


def test_sparse_depth_downsample_preserves_isolated_measurements() -> None:
    depth = torch.zeros(8, 8, dtype=torch.float32).numpy()
    depth[1, 1] = 4.0
    depth[3, 3] = 3.0
    depth[5, 5] = 2.0
    depth[7, 7] = 1.0

    resized = resize_depth_preserve_samples(depth, (4, 4))

    assert (resized > 0).sum() == 4
    assert torch.equal(
        torch.from_numpy(resized).diag(), torch.tensor([4.0, 3.0, 2.0, 1.0])
    )


def test_depth_downsample_keeps_nearest_surface_on_collision() -> None:
    depth = torch.zeros(4, 4, dtype=torch.float32).numpy()
    depth[0, 0] = 4.0
    depth[0, 1] = 2.0

    resized = resize_depth_preserve_samples(depth, (2, 2))

    assert resized[0, 0] == 2.0
    assert (resized > 0).sum() == 1


def test_dense_depth_downsample_matches_nearest_resampling() -> None:
    depth = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8).numpy()

    resized = resize_depth_preserve_samples(depth, (4, 4))

    assert torch.equal(
        torch.from_numpy(resized),
        torch.from_numpy(depth)[::2, ::2],
    )


def test_keyless_evaluation_manifest_freezes_named_subset(tmp_path) -> None:
    manifest = tmp_path / "sharp.json"
    manifest.write_text('["002", "000"]')
    parser = SimpleNamespace(image_names=["000.png", "001.png", "002.png"])

    indices, source = resolve_evaluation_indices(
        parser,
        {"evaluation_manifest": {"path": str(manifest), "label": "sharp"}},
        tmp_path,
        hold=2,
    )

    assert indices == [0, 2]
    assert source == "manifest:sharp"


def test_sharp_json_only_does_not_promote_evaluation_views(tmp_path) -> None:
    sharp = tmp_path / "sharp.json"
    sharp.write_text('["001.png"]')
    parser = SimpleNamespace(image_names=["000.png", "001.png", "002.png"])

    names, sources, policy = resolve_sharp_supervision(
        parser,
        {
            "data_dir": str(tmp_path),
            "sharp_json": str(sharp),
            "sharp_supervision_policy": "sharp_json_only",
        },
        evaluation_indices=[0, 1, 2],
    )

    assert names == {"001"}
    assert sources == ["sharp_json"]
    assert policy == "sharp_json_only"


def test_default_sharp_policy_preserves_hold_frame_supervision(tmp_path) -> None:
    sharp = tmp_path / "sharp.json"
    sharp.write_text("[]")
    parser = SimpleNamespace(image_names=["000.png", "001.png", "002.png"])

    names, sources, policy = resolve_sharp_supervision(
        parser,
        {"data_dir": str(tmp_path), "sharp_json": str(sharp)},
        evaluation_indices=[0, 2],
    )

    assert names == {"000", "002"}
    assert sources == ["sharp_json", "evaluation_indices"]
    assert policy == "evaluation_is_sharp"


def test_kernel_diagnostics_stratify_sharp_and_non_sharp_views() -> None:
    objective = BlurAwareObjective(6, BlurAwareObjectiveConfig(kernel_size=3))
    kernels, rows = blur_kernel_statistics(
        objective,
        [f"frame_{index}" for index in range(6)],
        torch.tensor([True, True, True, False, False, False]),
    )
    selected = stratified_kernel_indices(rows, limit=4)

    assert kernels.shape == (6, 3, 3)
    assert sum(bool(rows[index]["known_sharp"]) for index in selected) == 2
    assert len(selected) == 4
    assert all(float(row["rms_radius_px"]) >= 0.0 for row in rows)


def test_virtual_camera_precrop_preserves_intrinsics() -> None:
    preprocessor = CameraPreprocessor(
        source_size=(832, 480),
        pre_crop=(96, 0, 96, 0),
        resize_size=(528, 400),
        crop=(8, 8, 8, 8),
        distortion=(),
        undistort_depth=False,
    )
    intrinsics = torch.tensor(
        [[677.17, 0.0, 422.63], [0.0, 677.3, 252.61], [0.0, 0.0, 1.0]]
    )
    transformed = preprocessor.transform_intrinsics(intrinsics)
    assert preprocessor.cropped_source_size == (640, 480)
    assert preprocessor.output_size == (512, 384)
    assert torch.allclose(
        transformed,
        torch.tensor(
            [
                [558.6653, 0.0, 261.4698],
                [0.0, 564.4167, 202.5083],
                [0.0, 0.0, 1.0],
            ]
        ),
        atol=1e-4,
    )


def test_tum_protocol_maps_stream_positions_to_raw_names(tmp_path) -> None:
    timestamps = [0.00, 0.02, 0.04, 0.08, 0.12]
    (tmp_path / "rgb.txt").write_text(
        "\n".join(f"{stamp:.2f} rgb/{index}.png" for index, stamp in enumerate(timestamps))
    )
    (tmp_path / "depth.txt").write_text(
        "\n".join(f"{stamp:.2f} depth/{index}.png" for index, stamp in enumerate(timestamps))
    )
    (tmp_path / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(f"{stamp:.2f} 0 0 0 0 0 0 1" for stamp in timestamps)
    )

    protocol = resolve_tum_i2slam_protocol(
        {
            "tum_dir": str(tmp_path),
            "frame_rate": 32.0,
            "association_max_dt": 0.08,
            "evaluation_stream_indices": [0, 2],
        }
    )

    assert protocol.optimization_names == ["000000", "000002", "000003", "000004"]
    assert protocol.evaluation_names == ["000000", "000003"]
    assert protocol.metadata["optimization_views"] == 4


def test_range_depth_is_converted_to_camera_z() -> None:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    measurement = torch.tensor([2.0, 2.0**0.5])
    x = torch.tensor([50, 150])
    y = torch.tensor([50, 50])

    z = depth_measurement_to_z(measurement, x, y, intrinsics, "range")

    assert torch.allclose(z, torch.tensor([2.0, 1.0]))
    assert torch.equal(
        depth_measurement_to_z(measurement, x, y, intrinsics, "z"), measurement
    )


def _cfg(**overrides) -> AdaptiveStrategyCfg:
    values = {
        "name": "adaptive",
        "do_densify": True,
        "do_prune": True,
        "do_opacity_reset": False,
        "cap_max": 1_500_000,
        "active_budget": 100_000,
        "support_conditioned_cap": True,
        "min_growth_fraction": 0.0,
        "max_growth_fraction": 0.5,
        "pause_refine_after_reset": 0,
        "refine_every": 10,
        "reset_every": 10**9,
        "refine_start_iter": 5,
        "refine_stop_iter": 80,
        "refine_scale2d_stop_iter": 0,
        "grow_grad2d": 0.0,
        "grow_scale3d": 0.0,
        "prune_scale3d": 0.1,
        "prune_scale2d": 0.15,
        "grow_scale2d": 0.05,
        "min_opacity": 0.005,
        "prune_zero_radii": False,
        "reduce_opacity": False,
        "reduce_factor": 0.0,
        "reduce_every": 0,
        "fallback_means_lr": 0.0,
    }
    values.update(overrides)
    return AdaptiveStrategyCfg(**values)


def _state(num_points: int, visible_fraction: float) -> VanillaStrategyState:
    state = VanillaStrategyState.initialize(num_points, torch.device("cpu"), 1.0)
    state.denom.fill_(1.0)
    state.last_visible_fraction = visible_fraction
    return state


def test_adaptive_selection_is_invariant_to_gradient_units() -> None:
    num_points = 100
    gradients = torch.linspace(0.01, 1.0, num_points)
    scales = torch.linspace(0.001, 0.1, num_points)[:, None].repeat(1, 3)

    clone_a, split_a = adaptive_growth_masks(
        _cfg(), gradients, scales, _state(num_points, 0.5)
    )
    clone_b, split_b = adaptive_growth_masks(
        _cfg(), gradients * 10_000.0, scales, _state(num_points, 0.5)
    )

    assert torch.equal(clone_a, clone_b)
    assert torch.equal(split_a, split_b)


def test_adaptive_growth_respects_online_capacity() -> None:
    num_points = 100
    cfg = _cfg(active_budget=30, cap_max=120)
    state = _state(num_points, 0.5)
    gradients = torch.ones(num_points)
    scales = torch.rand(num_points, 3)

    clone, split = adaptive_growth_masks(cfg, gradients, scales, state)

    # Uniform unresolved support expands the reference cap to the explicit
    # safety boundary, but the event still cannot exceed available capacity.
    assert state.last_base_cap == 60
    assert state.last_effective_cap == 120
    assert int(clone.sum() + split.sum()) == state.last_growth_budget
    assert 0 < state.last_growth_budget <= 20


def test_lower_visibility_allows_larger_scene_without_dataset_rules() -> None:
    num_points = 100
    cfg = _cfg(active_budget=50, cap_max=500)
    gradients = torch.linspace(0.01, 1.0, num_points)
    scales = torch.rand(num_points, 3)
    high_visibility = _state(num_points, 0.5)
    low_visibility = _state(num_points, 0.2)

    adaptive_growth_masks(cfg, gradients, scales, high_visibility)
    adaptive_growth_masks(cfg, gradients, scales, low_visibility)

    assert high_visibility.last_effective_cap < 500
    assert low_visibility.last_effective_cap == 500
    assert low_visibility.last_effective_cap > high_visibility.last_effective_cap
    assert low_visibility.last_capacity_pressure < high_visibility.last_capacity_pressure


def test_capacity_pressure_suppresses_unnecessary_growth() -> None:
    gradients = torch.linspace(0.01, 1.0, 100)
    scales = torch.rand(100, 3)
    roomy = _state(100, 0.1)
    crowded = _state(100, 0.5)
    cfg = _cfg(active_budget=50, cap_max=500)

    adaptive_growth_masks(cfg, gradients, scales, roomy)
    adaptive_growth_masks(cfg, gradients, scales, crowded)

    assert roomy.last_capacity_pressure < crowded.last_capacity_pressure
    assert roomy.last_growth_fraction >= crowded.last_growth_fraction
    assert roomy.last_effective_cap > crowded.last_effective_cap


def test_residual_capacity_competition_closes_sparse_scene_gap() -> None:
    num_points = 100
    gradients = torch.ones(num_points)
    scales = torch.rand(num_points, 3)
    state = _state(num_points, 1.0)

    clone, split = adaptive_growth_masks(
        _cfg(active_budget=10_000), gradients, scales, state
    )

    # Uniform residual support and one-percent pressure should request the
    # universal per-event maximum, without consulting a dataset label or a
    # gradient-magnitude threshold.
    assert state.last_support_fraction == 1.0
    assert abs(state.last_capacity_pressure - 100 / 1_500_000) < 1e-12
    assert state.last_growth_fraction == 0.5
    assert int(clone.sum() + split.sum()) == 50


def test_broad_residual_support_raises_capacity_without_dataset_identity() -> None:
    num_points = 100
    scales = torch.rand(num_points, 3)
    cfg = _cfg(active_budget=50, cap_max=10_000)
    concentrated = _state(num_points, 0.5)
    broad = _state(num_points, 0.5)
    concentrated_gradients = torch.full((num_points,), 1e-6)
    concentrated_gradients[0] = 1.0

    adaptive_growth_masks(
        cfg, concentrated_gradients, scales, concentrated
    )
    adaptive_growth_masks(cfg, torch.ones(num_points), scales, broad)

    assert broad.last_demand_multiplier > concentrated.last_demand_multiplier
    assert broad.last_effective_cap > concentrated.last_effective_cap


def test_legacy_capacity_mode_is_an_exact_visibility_only_rollback() -> None:
    num_points = 100
    state = _state(num_points, 0.5)
    adaptive_growth_masks(
        _cfg(
            active_budget=30,
            cap_max=120,
            support_conditioned_cap=False,
        ),
        torch.ones(num_points),
        torch.rand(num_points, 3),
        state,
    )

    assert state.last_base_cap == 60
    assert state.last_demand_multiplier == 1.0
    assert state.last_effective_cap == num_points
    assert state.last_growth_budget == 0


def test_disabled_reward_is_an_exact_adaptive_selection_rollback() -> None:
    num_points = 100
    gradients = torch.linspace(0.01, 1.0, num_points)
    scales = torch.rand(num_points, 3)
    baseline = _state(num_points, 0.5)
    staged = _state(num_points, 0.5)
    set_adaptive_densification_feedback(
        staged,
        {
            "revision": 1,
            "probe_psnr": 99.0,
            "probe_surplus": 1.0,
            "has_surplus": True,
        },
    )

    baseline_masks = adaptive_growth_masks(
        _cfg(reward_conditioned=False), gradients, scales, baseline
    )
    staged_masks = adaptive_growth_masks(
        _cfg(reward_conditioned=False), gradients, scales, staged
    )

    assert torch.equal(baseline_masks[0], staged_masks[0])
    assert torch.equal(baseline_masks[1], staged_masks[1])
    assert baseline.last_effective_cap == staged.last_effective_cap
    assert baseline.last_growth_budget == staged.last_growth_budget
    assert staged.last_action_factor == 1.0


def test_delayed_probe_reward_modulates_growth_and_capacity() -> None:
    num_points = 100
    gradients = torch.linspace(0.01, 1.0, num_points)
    scales = torch.rand(num_points, 3)
    cfg = _cfg(
        active_budget=50,
        cap_max=1000,
        reward_conditioned=True,
        reward_ema_decay=0.5,
    )

    def run_transition(psnr: float, surplus: float) -> VanillaStrategyState:
        state = _state(num_points, 0.5)
        for revision, current_psnr, current_surplus in (
            (1, 30.0, 0.00),
            (2, 30.5, 0.05),
            (3, psnr, surplus),
        ):
            set_adaptive_densification_feedback(
                state,
                {
                    "revision": revision,
                    "probe_psnr": current_psnr,
                    "probe_surplus": current_surplus,
                    "has_surplus": True,
                },
            )
            adaptive_growth_masks(cfg, gradients, scales, state)
        return state

    improved = run_transition(31.5, 0.15)
    degraded = run_transition(29.5, -0.05)

    assert improved.last_reward_used is True
    assert degraded.last_reward_used is True
    assert improved.last_densification_reward > degraded.last_densification_reward
    assert improved.last_action_factor > 1.0
    assert degraded.last_action_factor < 1.0
    assert (
        improved.last_rewarded_support_fraction
        > degraded.last_rewarded_support_fraction
    )
    assert improved.last_effective_cap >= degraded.last_effective_cap
    assert improved.last_growth_budget >= degraded.last_growth_budget


def test_positive_probe_gain_remains_positive_after_convergence_slows() -> None:
    gradients = torch.linspace(0.01, 1.0, 100)
    scales = torch.rand(100, 3)
    state = _state(100, 0.5)
    cfg = _cfg(
        active_budget=50,
        cap_max=1000,
        reward_conditioned=True,
        reward_ema_decay=0.5,
    )

    for revision, psnr, surplus in (
        (1, 30.0, 0.00),
        (2, 31.0, 0.10),
        (3, 31.2, 0.12),
    ):
        set_adaptive_densification_feedback(
            state,
            {
                "revision": revision,
                "probe_psnr": psnr,
                "probe_surplus": surplus,
                "has_surplus": True,
            },
        )
        adaptive_growth_masks(cfg, gradients, scales, state)

    # Convergence slowed from +1.0 dB to +0.2 dB, but direct delayed utility
    # still records the real positive fixed-probe transition.
    assert abs(state.last_probe_psnr_delta - 0.2) < 1e-6
    assert abs(state.last_probe_surplus_delta - 0.02) < 1e-6
    assert state.last_quality_reward > 0.0
    assert state.last_action_factor > 1.0


def test_probe_surplus_alone_can_reward_a_densification_action() -> None:
    gradients = torch.linspace(0.01, 1.0, 100)
    scales = torch.rand(100, 3)
    state = _state(100, 0.5)
    cfg = _cfg(
        active_budget=50,
        cap_max=1000,
        reward_conditioned=True,
        reward_ema_decay=0.5,
    )

    for revision, surplus in ((1, 0.0), (2, 0.1), (3, 0.3)):
        set_adaptive_densification_feedback(
            state,
            {
                "revision": revision,
                "probe_psnr": 30.0,
                "probe_surplus": surplus,
                "has_surplus": True,
            },
        )
        adaptive_growth_masks(cfg, gradients, scales, state)

    assert state.last_probe_psnr_delta == 0.0
    assert abs(state.last_probe_surplus_delta - 0.2) < 1e-6
    assert state.last_quality_reward > 0.0
    assert state.last_action_factor > 1.0


def test_reward_history_survives_optimizer_phase_restart() -> None:
    gradients = torch.linspace(0.01, 1.0, 100)
    scales = torch.rand(100, 3)
    cfg = _cfg(reward_conditioned=True)
    before = _state(100, 0.5)
    for revision, psnr, surplus in (
        (1, 30.0, 0.0),
        (2, 30.5, 0.05),
    ):
        set_adaptive_densification_feedback(
            before,
            {
                "revision": revision,
                "probe_psnr": psnr,
                "probe_surplus": surplus,
                "has_surplus": True,
            },
        )
        adaptive_growth_masks(cfg, gradients, scales, before)

    after = _state(100, 0.5)
    transfer_adaptive_reward_state(before, after)
    set_adaptive_densification_feedback(
        after,
        {
            "revision": 3,
            "probe_psnr": 31.5,
            "probe_surplus": 0.15,
            "has_surplus": True,
        },
    )
    adaptive_growth_masks(cfg, gradients, scales, after)

    assert after.last_reward_used is True
    assert after.last_probe_psnr_delta == 1.0
    assert abs(after.last_probe_surplus_delta - 0.1) < 1e-6
    assert after.last_action_factor > 1.0
