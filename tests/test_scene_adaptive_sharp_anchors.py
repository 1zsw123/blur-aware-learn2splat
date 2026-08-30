from experiments.blur_aware_cross_dataset.scene_adaptive_sharp_anchors import (
    discover_sharp_anchors,
)


def test_mixture_discovers_high_quality_population_without_fixed_ratio() -> None:
    rows = [
        {"name": f"low_{index}", "nima_koniq": score}
        for index, score in enumerate([0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    ] + [
        {"name": f"high_{index}", "nima_koniq": score}
        for index, score in enumerate([0.52, 0.55, 0.58])
    ]

    result = discover_sharp_anchors(rows)

    assert result["selected_frames"] == ["high_2", "high_1", "high_0"]
    assert result["selected_count"] == 3
    assert result["selected_percent"] == 100.0 / 3.0


def test_mixture_fails_closed_for_constant_scores() -> None:
    rows = [
        {"name": f"frame_{index}", "nima_koniq": 0.2} for index in range(8)
    ]

    try:
        discover_sharp_anchors(rows)
    except ValueError as error:
        assert "non-constant" in str(error)
    else:
        raise AssertionError("constant scene scores must fail closed")
