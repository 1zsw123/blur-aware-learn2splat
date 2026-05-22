"""Benchmark COLMAP binary parsing vs .npz cache loading.

Usage:
    python src/scripts/benchmark_colmap_loading.py --root <path/to/scenes> [--scenes N] [--repeats R] [--normalize]

For each sampled scene the script measures:
  - Parser (raw .bin)  : full SceneManager + pose processing time
  - npz (cached)       : np.load() time after the cache has been written

Results are printed as a table and summary statistics.
"""
import argparse
import sys
import time
import os
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optgs.dataset.colmap.utils import Parser


# ── helpers ──────────────────────────────────────────────────────────────────

def npz_path(scene_dir: Path, normalize: bool) -> Path:
    suffix = "_norm" if normalize else ""
    return scene_dir / f"colmap_points_cache{suffix}.npz"


def find_scene_dirs(root: Path) -> list[Path]:
    scenes = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sparse = child / "sparse" / "0"
        if not sparse.exists():
            sparse = child / "sparse"
        if sparse.exists():
            scenes.append(child)
    return scenes


def time_bin(scene_dir: Path, normalize: bool) -> float:
    """Time a full Parser (raw COLMAP binary) parse."""
    t0 = time.perf_counter()
    parser = Parser(
        data_dir=str(scene_dir),
        factor=1,
        normalize=normalize,
        load_images=False,
        dl3dv_settings=False,
        verbose=False,
    )
    _ = parser.points, parser.points_rgb, parser.camtoworlds
    return time.perf_counter() - t0


def ensure_npz(scene_dir: Path, normalize: bool) -> Path:
    """Write the .npz cache if it doesn't exist (or is corrupt), return its path."""
    p = npz_path(scene_dir, normalize)

    # Delete corrupt/empty files before attempting to create.
    if p.exists():
        try:
            data = np.load(p)
            _ = data["points"], data["points_rgb"], data["camtoworlds"]
            return p  # healthy, nothing to do
        except Exception:
            print(f"  WARNING: corrupt cache found at {p}, deleting and re-creating…")
            p.unlink(missing_ok=True)

    print(f"  Creating .npz cache for {scene_dir.name}…", end="", flush=True)
    parser = Parser(
        data_dir=str(scene_dir),
        factor=1,
        normalize=normalize,
        load_images=False,
        dl3dv_settings=False,
        verbose=False,
    )
    # NOTE: np.savez_compressed auto-appends ".npz" if the path doesn't end
    # with it — so the temp file must already carry the .npz suffix, otherwise
    # savez writes to "<tmp>.npz" while tmp_path points to the empty "<tmp>".
    tmp_fd, tmp_path = tempfile.mkstemp(dir=scene_dir, suffix=".npz")
    os.close(tmp_fd)
    try:
        np.savez_compressed(
            tmp_path,
            points=parser.points,
            points_rgb=parser.points_rgb,
            camtoworlds=parser.camtoworlds,
        )
        # Verify it's readable before promoting to the final path.
        data = np.load(tmp_path, allow_pickle=False)
        _ = data["points"], data["points_rgb"], data["camtoworlds"]
        os.replace(tmp_path, p)
        print(" done.")
    except Exception:
        print(f"  ERROR creating .npz cache for {scene_dir.name}.", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return p


def time_npz(scene_dir: Path, normalize: bool) -> float:
    """Time loading from the .npz cache."""
    p = npz_path(scene_dir, normalize)
    t0 = time.perf_counter()
    data = np.load(p)
    _ = data["points"], data["points_rgb"], data["camtoworlds"]
    return time.perf_counter() - t0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path,
                        help="Root directory containing one sub-dir per scene.")
    parser.add_argument("--scenes", type=int, default=10,
                        help="Number of scenes to benchmark (default: 10).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeat each timing N times and take the median (default: 3).")
    parser.add_argument("--normalize", action="store_true",
                        help="Use normalize=True (matches normalize_world_space: true in config).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    all_scenes = find_scene_dirs(root)
    if not all_scenes:
        print(f"No COLMAP scenes found under {root}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    n = min(args.scenes, len(all_scenes))
    scenes = [all_scenes[i] for i in rng.choice(len(all_scenes), size=n, replace=False)]

    print(f"Benchmarking {n} scenes  (repeats={args.repeats}, normalize={args.normalize})\n")

    # Pre-create all .npz caches so the write cost doesn't pollute the timing.
    print("Pre-creating .npz caches (if missing)…")
    good_scenes = []
    for s in scenes:
        try:
            ensure_npz(s, args.normalize)
            good_scenes.append(s)
        except Exception as e:
            print(f"  SKIP {s.name}: {e}", file=sys.stderr)
    print(f"Done. {len(good_scenes)}/{len(scenes)} scenes OK.\n")

    if not good_scenes:
        print("No valid scenes to benchmark.", file=sys.stderr)
        sys.exit(1)

    col_w = max(len(s.name) for s in good_scenes)
    header = f"{'Scene':<{col_w}}  {'bin (s)':>10}  {'npz (s)':>10}  {'speedup':>10}"
    print(header)
    print("-" * len(header))

    bin_times, npz_times, speedups = [], [], []

    for scene in good_scenes:
        print(f"  timing {scene.name}…", end="", flush=True)
        try:
            b = np.median([time_bin(scene, args.normalize) for _ in range(args.repeats)])
            z = np.median([time_npz(scene, args.normalize) for _ in range(args.repeats)])
        except Exception as e:
            print(f"\r  SKIP {scene.name}: {e}", file=sys.stderr)
            continue
        sp = b / z if z > 0 else float("inf")

        bin_times.append(b)
        npz_times.append(z)
        speedups.append(sp)

        print(f"\r{scene.name:<{col_w}}  {b:>10.3f}  {z:>10.4f}  {sp:>9.1f}x")

    print("-" * len(header))
    if not bin_times:
        print("No scenes were successfully benchmarked.")
        sys.exit(1)
    print(f"{'MEAN':<{col_w}}  {np.mean(bin_times):>10.3f}  {np.mean(npz_times):>10.4f}  {np.mean(speedups):>9.1f}x")
    print(f"{'MEDIAN':<{col_w}}  {np.median(bin_times):>10.3f}  {np.median(npz_times):>10.4f}  {np.median(speedups):>9.1f}x")
    print(f"{'MAX':<{col_w}}  {np.max(bin_times):>10.3f}  {np.max(npz_times):>10.4f}  {np.max(speedups):>9.1f}x")


if __name__ == "__main__":
    main()

