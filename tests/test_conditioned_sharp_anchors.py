import numpy as np

from experiments.blur_aware_cross_dataset.prepare_deblurnerf_conditioned_sharp_anchors import (
    fit_conditioned_split,
    fit_nima_only_split,
    robust_normalize,
)


def test_robust_normalize_uses_scene_median_and_mad():
    normalized, median, mad = robust_normalize(np.array([1.0, 2.0, 3.0, 20.0]))
    assert median == 2.5
    assert mad == 1.0
    np.testing.assert_allclose(normalized, [-1.5, -0.5, 0.5, 17.5])


def test_conditioned_split_selects_high_nima_low_restoration_demand():
    rng = np.random.default_rng(7)
    blurry_nima = rng.normal(0.48, 0.015, 30)
    blurry_gain = rng.normal(0.35, 0.025, 30)
    sharp_nima = rng.normal(0.72, 0.015, 12)
    sharp_gain = rng.normal(0.02, 0.02, 12)
    result = fit_conditioned_split(
        [f"{index:03d}" for index in range(42)],
        np.concatenate((blurry_nima, sharp_nima)),
        np.concatenate((blurry_gain, sharp_gain)),
    )
    assert result["status"] == "PASS"
    selected = result["selected_mask"]
    assert selected[:30].sum() == 0
    assert selected[30:].sum() == 12


def test_nima_only_split_selects_high_quality_component():
    rng = np.random.default_rng(11)
    scores = np.concatenate((rng.normal(0.45, 0.01, 24), rng.normal(0.72, 0.01, 8)))
    result = fit_nima_only_split(scores)
    assert result["status"] == "PASS"
    assert result["selected_mask"][:24].sum() == 0
    assert result["selected_mask"][24:].sum() == 8
