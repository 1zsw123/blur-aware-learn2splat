# Blur-aware LeGS local-state alignment (S1)

## Problem

The main reconstruction gradient and ADC candidate gate already used the joint
EVSSM-BPN, RAW-BPN, and Laplacian-surplus objective. However, LeGS still built
its 11-D per-Gaussian policy state and delayed local sensitivity reward from the
plain target loss. The global blur observation was duplicated over all
Gaussians, so the policy could not assign a blur-aware local action reliably.

## Fix

- Build the local LeGS gradient state from the joint blur-aware objective.
- Treat that state query as read-only: do not update BPN parameters or surplus
  reliability statistics a second time.
- Credit every changed action in a successful structural event. Net primitive
  count remains a diagnostic and no longer erases balanced birth/prune credit.
- When all views have authoritative direct supervision, bypass local blur
  conditioning, global blur features, blur probe rendering, and blur reward.
  This reduces exactly to the original LeGS learning problem.
- Keep `--no-legs-local-objective` as the target-only ablation.

## Smoke evidence

All runs use seed 20260824, 3K steps, evaluations at 1K/2K/3K, FastGS,
Learn2Splat projected optimization, coupled dual BPN, and identical data.

| Scene | Target-only local state (3K) | Joint local state (3K) | Delta |
|---|---:|---:|---:|
| motion_blurcoffee | 42.920 | 43.121 | +0.201 dB |
| defocus_cisco | 31.310 | 31.392 | +0.082 dB |

TUM `fr2_xyz` contains authoritative direct supervision for every benchmark
view, so blur conditioning must be inactive. The final direct-only short-circuit
run reached 25.508 dB / 0.8498 SSIM at 3K, while an original-LeGS reference run
reached 25.325 dB / 0.8474. This difference is within stochastic CUDA/policy
variation and is not claimed as blur-module improvement. The relevant result is
that the blur branch contributes zero state and zero reward in this case.

## Verification

`PYTHONPATH=$PWD .../learn2splat-py312/bin/pytest -q`: 79 passed.
