"""Dataset protocol adapters kept outside the learned reconstruction method.

These functions only reproduce benchmark frame streams and splits. They never
feed a dataset identity or a benchmark metric into Learn2Splat or its capacity
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResolvedFrameProtocol:
    optimization_names: list[str]
    evaluation_names: list[str]
    metadata: dict[str, object]


def _parse_tum_list(path: Path, *, skip_rows: int = 0) -> np.ndarray:
    rows: list[list[str]] = []
    for line in path.read_text().splitlines()[skip_rows:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(stripped.split())
    if not rows:
        raise RuntimeError(f"no usable TUM rows in {path}")
    return np.asarray(rows, dtype=np.str_)


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match ``np.argmin(abs(reference - q))`` for sorted timestamps."""
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def resolve_tum_i2slam_protocol(spec: dict) -> ResolvedFrameProtocol:
    """Reproduce Unblur-SLAM's ``TUM_RGB.loadtum(frame_rate=32)`` stream.

    Unblur-SLAM's published evaluation numbers are positions in this associated
    and temporally filtered stream, not raw RGB filenames. The returned names
    use the raw-row index naming contract of our immutable COLMAP scene builder.
    """
    tum_dir = Path(spec["tum_dir"])
    frame_rate = float(spec.get("frame_rate", 32.0))
    max_dt = float(spec.get("association_max_dt", 0.08))
    evaluation_stream_indices = [
        int(value) for value in spec["evaluation_stream_indices"]
    ]
    if frame_rate <= 0.0 or max_dt <= 0.0:
        raise ValueError("TUM protocol frame_rate and association_max_dt must be positive")

    rgb = _parse_tum_list(tum_dir / "rgb.txt")
    depth = _parse_tum_list(tum_dir / "depth.txt")
    pose = _parse_tum_list(tum_dir / "groundtruth.txt", skip_rows=1)
    rgb_t = rgb[:, 0].astype(np.float64)
    depth_t = depth[:, 0].astype(np.float64)
    pose_t = pose[:, 0].astype(np.float64)

    depth_match = _nearest_indices(rgb_t, depth_t)
    pose_match = _nearest_indices(rgb_t, pose_t)
    associated = np.flatnonzero(
        (np.abs(depth_t[depth_match] - rgb_t) < max_dt)
        & (np.abs(pose_t[pose_match] - rgb_t) < max_dt)
    )
    if associated.size == 0:
        raise RuntimeError(f"TUM protocol found no associated frames in {tum_dir}")

    # This intentionally mirrors the official loader's comparison against the
    # last accepted timestamp, including its strict greater-than condition.
    selected_positions = [0]
    min_period = 1.0 / frame_rate
    for position in range(1, len(associated)):
        previous = associated[selected_positions[-1]]
        current = associated[position]
        if rgb_t[current] - rgb_t[previous] > min_period:
            selected_positions.append(position)
    stream_raw_indices = associated[np.asarray(selected_positions)].tolist()

    invalid_eval = [
        index
        for index in evaluation_stream_indices
        if index < 0 or index >= len(stream_raw_indices)
    ]
    if invalid_eval:
        raise IndexError(
            f"evaluation indices outside {len(stream_raw_indices)}-frame TUM stream: "
            f"{invalid_eval}"
        )
    optimization_names = [f"{index:06d}" for index in stream_raw_indices]
    evaluation_names = [
        optimization_names[index] for index in evaluation_stream_indices
    ]
    return ResolvedFrameProtocol(
        optimization_names=optimization_names,
        evaluation_names=evaluation_names,
        metadata={
            "type": "tum_i2slam_32hz",
            "tum_dir": str(tum_dir),
            "raw_rgb_rows": int(len(rgb)),
            "associated_rows": int(len(associated)),
            "optimization_views": int(len(optimization_names)),
            "evaluation_views": int(len(evaluation_names)),
            "frame_rate": frame_rate,
            "association_max_dt": max_dt,
            "evaluation_stream_indices": evaluation_stream_indices,
            "evaluation_raw_names": evaluation_names,
        },
    )


def resolve_frame_protocol(spec: dict) -> ResolvedFrameProtocol:
    protocol_type = spec.get("type")
    if protocol_type == "tum_i2slam_32hz":
        return resolve_tum_i2slam_protocol(spec)
    raise ValueError(f"unsupported frame protocol {protocol_type!r}")
