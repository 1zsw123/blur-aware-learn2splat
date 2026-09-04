#!/usr/bin/env python3
"""Compare ExBlur restoration teachers against non-evaluation sharp frames."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DATA_ROOT = Path("/srv2/szha0669/blur_slam_exp/data/exblurnerf/exblur_release")
EVSSM_ROOT = Path("/srv2/szha0669/blur_slam_exp/data/evssm_deblurred_exblurnerf")
TURTLE_ROOT = Path("/srv2/szha0669/blur_slam_exp/data/turtle_stage1_step024000_exblur8_s1")
MANIFEST_ROOT = Path("/srv2/szha0669/blur_slam_exp/outputs/logs/exblur8_official_protocol_s1")
SCENES = ("bench", "camellia", "dragon", "jars", "jars2", "postbox", "stone_lantern", "sunflowers")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if prediction.shape != target.shape:
        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)
    return (
        float(peak_signal_noise_ratio(target, prediction, data_range=255)),
        float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
    )


def main() -> None:
    report: dict[str, object] = {"scope": "non_official_test_frames", "scenes": {}}
    aggregate: dict[str, list[tuple[float, float]]] = {name: [] for name in ("raw", "evssm", "turtle")}

    for scene in SCENES:
        holds = set(json.loads((MANIFEST_ROOT / f"{scene}_official_test.json").read_text()))
        sharp_dir = DATA_ROOT / scene / "images_sharp"
        source_dirs = {
            "raw": DATA_ROOT / scene / "images",
            "evssm": EVSSM_ROOT / scene,
            "turtle": TURTLE_ROOT / f"exblur_{scene}",
        }
        names = sorted(path.name for path in sharp_dir.glob("*.png") if path.name not in holds)
        scene_result: dict[str, object] = {"frames": len(names), "metrics": {}}
        for method, source_dir in source_dirs.items():
            values = [metrics(read_rgb(source_dir / name), read_rgb(sharp_dir / name)) for name in names]
            aggregate[method].extend(values)
            scene_result["metrics"][method] = {
                "psnr": float(np.mean([value[0] for value in values])),
                "ssim": float(np.mean([value[1] for value in values])),
            }
        report["scenes"][scene] = scene_result

    report["aggregate_frame_weighted"] = {
        method: {
            "frames": len(values),
            "psnr": float(np.mean([value[0] for value in values])),
            "ssim": float(np.mean([value[1] for value in values])),
        }
        for method, values in aggregate.items()
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
