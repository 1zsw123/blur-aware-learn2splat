#!/usr/bin/env python3
"""Discover scene-relative sharp anchors from NIMA scores without split labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


def discover_sharp_anchors(
    rows: list[dict[str, object]],
    *,
    min_bic_gain: float = 0.0,
    min_separation: float = 2.0,
) -> dict[str, object]:
    """Fit a two-population scene model and return its high-quality component."""
    names = np.asarray([str(row["name"]) for row in rows])
    scores = np.asarray([float(row["nima_koniq"]) for row in rows], dtype=float)
    if len(rows) < 4 or len(set(names)) != len(names):
        raise ValueError("NIMA rows must contain at least four unique frame names")
    if not np.isfinite(scores).all() or len(np.unique(scores)) < 2:
        raise ValueError("NIMA scores must be finite and non-constant")

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad <= np.finfo(float).eps:
        raise ValueError("NIMA score MAD is zero; scene split is not identifiable")
    normalized = ((scores - median) / mad).reshape(-1, 1)

    mixture = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=50,
        random_state=0,
        reg_covar=1e-6,
    ).fit(normalized)
    one_component = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-6,
    ).fit(normalized)

    means = mixture.means_.ravel()
    variances = mixture.covariances_.reshape(-1)
    sharp_component = int(np.argmax(means))
    other_component = 1 - sharp_component
    separation = abs(means[sharp_component] - means[other_component]) / math.sqrt(
        (variances[sharp_component] + variances[other_component]) / 2.0
    )
    bic_gain = float(one_component.bic(normalized) - mixture.bic(normalized))
    if bic_gain <= min_bic_gain or separation < min_separation:
        raise RuntimeError(
            "NIMA sharp-anchor split is not identifiable: "
            f"bic_gain={bic_gain:.4f}, separation={separation:.4f}"
        )

    sharp_probability = mixture.predict_proba(normalized)[:, sharp_component]
    selected_mask = sharp_probability > 0.5
    if not selected_mask.any() or selected_mask.all():
        raise RuntimeError("NIMA mixture produced an empty or full sharp set")
    threshold = float(scores[selected_mask].min())
    if bool((scores[~selected_mask] >= threshold).any()):
        raise RuntimeError("NIMA mixture does not define a monotonic upper-tail split")

    selected_rows = sorted(
        (
            {
                "name": str(name),
                "nima_koniq": float(score),
                "sharp_probability": float(probability),
            }
            for name, score, probability in zip(
                names[selected_mask],
                scores[selected_mask],
                sharp_probability[selected_mask],
                strict=True,
            )
        ),
        key=lambda row: (-row["nima_koniq"], row["name"]),
    )
    return {
        "frame_count": len(rows),
        "selected_count": len(selected_rows),
        "selected_percent": 100.0 * len(selected_rows) / len(rows),
        "threshold_score": threshold,
        "selected_frames": [row["name"] for row in selected_rows],
        "selected": selected_rows,
        "median": median,
        "mad": mad,
        "component_separation": float(separation),
        "bic_gain_two_vs_one": bic_gain,
        "posterior_threshold": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = discover_sharp_anchors(json.loads(args.scores.read_text()))
    args.anchors.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.anchors.write_text(json.dumps(result["selected_frames"], indent=2) + "\n")
    args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
