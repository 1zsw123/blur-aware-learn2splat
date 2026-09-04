#!/usr/bin/env python3
"""Prepare hold-blind Deblur-NeRF sharp anchors from scene-relative evidence.

The selector deliberately does not read hold/test identities.  It combines a
scene-normalized NIMA score with the relative sharpening demand observed from
the fixed EVSSM restoration, then separates two latent frame populations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


EPS = 1e-8


def robust_normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if not np.isfinite(values).all() or mad <= np.finfo(float).eps:
        raise ValueError("feature must be finite and have nonzero MAD")
    return (values - median) / mad, median, mad


def multiscale_laplacian_sharpness(image: np.ndarray) -> float:
    """Measure structured high frequencies while suppressing sensor noise."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR color image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    responses = []
    for sigma in (1.0, 2.0):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        laplacian = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
        responses.append(float(np.mean(np.abs(laplacian))))
    return float(np.mean(responses))


def image_index(directory: Path) -> dict[str, Path]:
    allowed = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in allowed
    )
    if not paths:
        raise RuntimeError(f"no images found in {directory}")
    return {path.stem: path for path in paths}


def resolve(index: dict[str, Path], name: str, directory: Path) -> Path:
    stem = Path(name).stem
    for candidate in (stem, stem.zfill(3), stem.zfill(4), stem.zfill(6)):
        if candidate in index:
            return index[candidate]
    raise FileNotFoundError(f"no image matching {name!r} in {directory}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_file_stem(scene: str) -> str:
    if scene.startswith("motion_"):
        return scene.removeprefix("motion_")
    if scene.startswith("defocus_"):
        return "defocus" + scene.removeprefix("defocus_")
    raise ValueError(f"unsupported scene key: {scene}")


def fit_conditioned_split(
    names: list[str], nima: np.ndarray, restoration_gain: np.ndarray
) -> dict[str, object]:
    if len(names) < 8 or len(set(names)) != len(names):
        raise ValueError("scene needs at least eight unique frames")
    z_nima, nima_median, nima_mad = robust_normalize(nima)
    z_gain, gain_median, gain_mad = robust_normalize(restoration_gain)
    features = np.column_stack((z_nima, z_gain))

    one = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-5,
    ).fit(features)
    two = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=50,
        random_state=0,
        reg_covar=1e-5,
    ).fit(features)

    # A sharp input should have relatively high perceptual quality and require
    # relatively little additional sharpening from the restoration teacher.
    component_scores = two.means_[:, 0] - 0.5 * two.means_[:, 1]
    sharp_component = int(np.argmax(component_scores))
    posterior = two.predict_proba(features)[:, sharp_component]
    selected = posterior > 0.5

    delta = two.means_[0] - two.means_[1]
    pooled_covariance = 0.5 * (two.covariances_[0] + two.covariances_[1])
    separation = float(
        math.sqrt(delta @ np.linalg.pinv(pooled_covariance) @ delta)
    )
    bic_gain = float(one.bic(features) - two.bic(features))
    selected_count = int(selected.sum())
    failures = []
    if bic_gain <= 0.0:
        failures.append("two_components_not_preferred_by_bic")
    if separation < 1.5:
        failures.append("component_separation_below_1p5")
    if selected_count == 0 or selected_count == len(names):
        failures.append("empty_or_full_sharp_component")
    if selected_count:
        selected_nima = float(nima[selected].mean())
        rejected_nima = float(nima[~selected].mean()) if (~selected).any() else float("nan")
        if (~selected).any() and selected_nima <= rejected_nima:
            failures.append("selected_component_not_higher_mean_nima")

    return {
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "failures": failures,
        "selected_mask": selected,
        "posterior": posterior,
        "z_nima": z_nima,
        "z_restoration_gain": z_gain,
        "nima_median": nima_median,
        "nima_mad": nima_mad,
        "restoration_gain_median": gain_median,
        "restoration_gain_mad": gain_mad,
        "bic_gain_two_vs_one": bic_gain,
        "component_separation": separation,
        "sharp_component": sharp_component,
        "component_weights": two.weights_.tolist(),
        "component_means_z_nima_z_gain": two.means_.tolist(),
        "component_covariances": two.covariances_.tolist(),
    }


def fit_nima_only_split(nima: np.ndarray) -> dict[str, object]:
    """Fit the originally proposed one-dimensional scene-relative NIMA GMM."""
    z_nima, median, mad = robust_normalize(nima)
    features = z_nima.reshape(-1, 1)
    one = GaussianMixture(
        n_components=1, n_init=10, random_state=0, reg_covar=1e-5
    ).fit(features)
    two = GaussianMixture(
        n_components=2, n_init=50, random_state=0, reg_covar=1e-5
    ).fit(features)
    sharp_component = int(np.argmax(two.means_.ravel()))
    posterior = two.predict_proba(features)[:, sharp_component]
    selected = posterior > 0.5
    variances = two.covariances_.reshape(-1)
    means = two.means_.ravel()
    separation = float(
        abs(means[0] - means[1]) / math.sqrt(0.5 * variances.sum())
    )
    bic_gain = float(one.bic(features) - two.bic(features))
    failures = []
    if bic_gain <= 0.0:
        failures.append("two_components_not_preferred_by_bic")
    if separation < 2.0:
        failures.append("component_separation_below_2p0")
    if int(selected.sum()) in (0, len(nima)):
        failures.append("empty_or_full_sharp_component")
    return {
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "failures": failures,
        "selected_mask": selected,
        "posterior": posterior,
        "median": median,
        "mad": mad,
        "bic_gain_two_vs_one": bic_gain,
        "component_separation": separation,
        "sharp_component": sharp_component,
        "component_weights": two.weights_.tolist(),
        "component_means_z_nima": means.tolist(),
        "component_variances": variances.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--nima-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configs = json.loads(args.scene_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    generated_configs = {}
    reports = {}
    summary_rows = []

    for scene, base_cfg in configs.items():
        stem = scene_file_stem(scene)
        score_path = args.nima_root / f"{stem}_nima_koniq_scores.json"
        old_path = args.nima_root / f"{stem}_nima06_sharp_frames.json"
        score_rows = json.loads(score_path.read_text())
        old_names_source = {
            Path(name).stem for name in json.loads(old_path.read_text())
        }

        raw_dir = Path(base_cfg["raw_dir"])
        evssm_dir = Path(base_cfg["evssm_dir"])
        raw_index = image_index(raw_dir)
        evssm_index = image_index(evssm_dir)
        frame_rows = []
        excluded_stale_score_rows = []
        for score_row in score_rows:
            name = Path(str(score_row["name"])).stem
            try:
                raw_path = resolve(raw_index, name, raw_dir)
                evssm_path = resolve(evssm_index, name, evssm_dir)
            except FileNotFoundError as error:
                excluded_stale_score_rows.append(
                    {"name": name, "reason": str(error)}
                )
                continue
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            restored = cv2.imread(str(evssm_path), cv2.IMREAD_COLOR)
            if raw is None or restored is None:
                raise RuntimeError(f"failed to read {raw_path} or {evssm_path}")
            if restored.shape[:2] != raw.shape[:2]:
                restored = cv2.resize(
                    restored,
                    (raw.shape[1], raw.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            raw_sharpness = multiscale_laplacian_sharpness(raw)
            restored_sharpness = multiscale_laplacian_sharpness(restored)
            frame_rows.append(
                {
                    "name": name,
                    "nima_koniq": float(score_row["nima_koniq"]),
                    "raw_multiscale_laplacian": raw_sharpness,
                    "evssm_multiscale_laplacian": restored_sharpness,
                    "restoration_gain_log": float(
                        math.log((restored_sharpness + EPS) / (raw_sharpness + EPS))
                    ),
                    "raw_path": str(raw_path),
                    "evssm_path": str(evssm_path),
                }
            )

        names = [row["name"] for row in frame_rows]
        valid_names = set(names)
        old_names = old_names_source & valid_names
        excluded_stale_nima06_names = sorted(old_names_source - valid_names)
        result = fit_conditioned_split(
            names,
            np.asarray([row["nima_koniq"] for row in frame_rows]),
            np.asarray([row["restoration_gain_log"] for row in frame_rows]),
        )
        nima_only = fit_nima_only_split(
            np.asarray([row["nima_koniq"] for row in frame_rows])
        )
        for index, row in enumerate(frame_rows):
            row["z_nima"] = float(result["z_nima"][index])
            row["z_restoration_gain"] = float(
                result["z_restoration_gain"][index]
            )
            row["sharp_posterior"] = float(result["posterior"][index])
            row["selected_conditioned_gmm"] = bool(
                result["selected_mask"][index]
            )
            row["selected_nima06"] = row["name"] in old_names
            row["nima_gmm_posterior"] = float(nima_only["posterior"][index])
            row["selected_nima_gmm"] = bool(nima_only["selected_mask"][index])

        new_names = {
            row["name"] for row in frame_rows if row["selected_conditioned_gmm"]
        }
        intersection = new_names & old_names
        union = new_names | old_names
        nima_gmm_names = {
            row["name"] for row in frame_rows if row["selected_nima_gmm"]
        }
        selected_path = args.output_dir / f"{stem}_conditioned_gmm_sharp_frames.json"
        weights_path = args.output_dir / f"{stem}_conditioned_gmm_weights_w10.json"
        posterior_path = args.output_dir / f"{stem}_conditioned_gmm_posteriors.json"
        selected_path.write_text(json.dumps(sorted(new_names), indent=2) + "\n")
        weights_path.write_text(
            json.dumps(
                {name: (10.0 if name in new_names else 1.0) for name in sorted(names)},
                indent=2,
            )
            + "\n"
        )
        posterior_path.write_text(json.dumps(frame_rows, indent=2) + "\n")
        nima_gmm_path = args.output_dir / f"{stem}_nima_gmm_sharp_frames.json"
        nima_gmm_path.write_text(json.dumps(sorted(nima_gmm_names), indent=2) + "\n")

        report = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "selected_mask",
                "posterior",
                "z_nima",
                "z_restoration_gain",
            }
        }
        report.update(
            {
                "scene": scene,
                "frame_count": len(names),
                "conditioned_gmm_count": len(new_names),
                "conditioned_gmm_percent": 100.0 * len(new_names) / len(names),
                "nima06_count": len(old_names),
                "nima06_percent": 100.0 * len(old_names) / len(names),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "conditioned_only": sorted(new_names - old_names),
                "nima06_only": sorted(old_names - new_names),
                "selected_path": str(selected_path),
                "weights_path": str(weights_path),
                "posterior_path": str(posterior_path),
                "score_source": str(score_path),
                "score_source_sha256": file_sha256(score_path),
                "nima06_source": str(old_path),
                "nima06_source_sha256": file_sha256(old_path),
                "nima_gmm": {
                    key: value
                    for key, value in nima_only.items()
                    if key not in {"selected_mask", "posterior"}
                },
                "nima_gmm_count": len(nima_gmm_names),
                "nima_gmm_percent": 100.0 * len(nima_gmm_names) / len(names),
                "nima_gmm_vs_nima06_intersection": len(nima_gmm_names & old_names),
                "nima_gmm_vs_nima06_jaccard": (
                    len(nima_gmm_names & old_names) / len(nima_gmm_names | old_names)
                    if nima_gmm_names | old_names
                    else 1.0
                ),
                "nima_gmm_selected_path": str(nima_gmm_path),
                "excluded_stale_score_rows": excluded_stale_score_rows,
                "excluded_stale_nima06_names": excluded_stale_nima06_names,
            }
        )
        reports[scene] = report

        cfg = dict(base_cfg)
        cfg["sharp_json"] = str(selected_path)
        cfg["sharp_weights_json"] = str(weights_path)
        cfg["sharp_supervision_policy"] = "sharp_json_only"
        cfg["evaluation_direct_supervision"] = False
        cfg["hold_blind_training"] = True
        cfg["sharp_anchor_discovery"] = {
            "method": "conditioned_two_component_gmm",
            "features": [
                "scene_robust_normalized_nima_koniq",
                "scene_robust_normalized_evssm_vs_raw_multiscale_laplacian_gain",
            ],
            "posterior_threshold": 0.5,
            "report_status": report["status"],
        }
        generated_configs[scene] = cfg
        summary_rows.append(
            {
                "scene": scene,
                "status": report["status"],
                "frames": len(names),
                "conditioned_gmm_count": len(new_names),
                "conditioned_gmm_percent": report["conditioned_gmm_percent"],
                "nima06_count": len(old_names),
                "nima06_percent": report["nima06_percent"],
                "nima_gmm_status": report["nima_gmm"]["status"],
                "nima_gmm_count": report["nima_gmm_count"],
                "nima_gmm_percent": report["nima_gmm_percent"],
                "nima_gmm_vs_nima06_jaccard": report[
                    "nima_gmm_vs_nima06_jaccard"
                ],
                "nima_gmm_bic_gain": report["nima_gmm"][
                    "bic_gain_two_vs_one"
                ],
                "nima_gmm_separation": report["nima_gmm"][
                    "component_separation"
                ],
                "nima_gmm_failures": ";".join(report["nima_gmm"]["failures"]),
                "intersection": len(intersection),
                "jaccard": report["jaccard"],
                "bic_gain": report["bic_gain_two_vs_one"],
                "separation": report["component_separation"],
                "failures": ";".join(report["failures"]),
            }
        )

    (args.output_dir / "scenes_conditioned_gmm.json").write_text(
        json.dumps(generated_configs, indent=2) + "\n"
    )
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    conditioned_failures = [
        scene for scene, report in reports.items() if report["status"] != "PASS"
    ]
    nima_gmm_failures = [
        scene
        for scene, report in reports.items()
        if report["nima_gmm"]["status"] != "PASS"
    ]
    (args.output_dir / "PREPROCESSING_STATUS.json").write_text(
        json.dumps(
            {
                "training_ready": False,
                "reason": (
                    "Neither scene-adaptive split passes its identifiability "
                    "gate on all 21 scenes; no training is authorized from this directory."
                ),
                "conditioned_gmm_failures": conditioned_failures,
                "nima_only_gmm_failures": nima_gmm_failures,
            },
            indent=2,
        )
        + "\n"
    )
    with (args.output_dir / "comparison_nima06.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
