---
title: Learn2Splat
emoji: 🪴
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive demo of the Learn2Splat learned 3DGS optimizer
---

# Learn2Splat — interactive demo

## Blur-aware cross-dataset research extension

This branch extends the official Learn2Splat Gaussian representation for
offline deblurring. The current tested pipeline combines:

- the released Learn2Splat optimizer followed by an objective-consistent Adam
  projection stage;
- BPN image-formation supervision and NIMA-sharp weighted view sampling;
- Scene-Adaptive Sharp Anchor Discovery, which fits a hold-blind scene-relative
  NIMA mixture and infers the sharp-anchor ratio instead of fixing a score or
  top percentage;
- a confidence-gated, multiscale Laplacian surplus objective that allows
  supported render sharpness to exceed the EVSSM initialization; and
- two capacity paths: a scale-free adaptive rollback whose delayed reward uses
  fixed training probes, and the exact LeGS per-Gaussian PPO candidate.

The stable rollback controller is the custom adaptive ADC described below.
The newest capacity candidate, `--adc legs`, transplants the released LeGS
mechanism into the Learn2Splat runtime: the pinned official FastGS CUDA
leave-one-out sensitivity, 11-D per-Gaussian state, keep/clone/split actor,
prune estimator, 50-step parent-child reward, and PPO update. It deliberately
does not reuse the older global/proxy `adapt` controller. The implementation,
equations, benchmark protocols, rollback flags, smoke-test table, and
reproduction commands are documented in
[`experiments/blur_aware_cross_dataset/README.md`](experiments/blur_aware_cross_dataset/README.md).
The selector equations, ExBlur 8-scene split audit, and label-leakage boundary
are documented separately in
[`SCENE_ADAPTIVE_SHARP_ANCHORS.md`](experiments/blur_aware_cross_dataset/SCENE_ADAPTIVE_SHARP_ANCHORS.md).

The separate experimental mode `--adc legs_blur` retains that exact local
LeGS mechanism and appends seven bounded scene-level blur features: EVSSM
reliability, Laplacian surplus, BPN kernel/mask statistics, and primitive
pressure. Its delayed reward combines fixed-training-probe PSNR/sharpness
improvement with the original sensitivity reward and a relative capacity
cost. A bias-free zero-initialized adapter preserves exact LeGS throughout the
2K warmup, then introduces the blur state linearly through 5K. Exact LeGS
remains available unchanged with `--adc legs`. The matched
Motion/Defocus/TUM smoke results and their current quality/capacity tradeoff
are recorded in
[`BLUR_CONDITIONED_LEGS_SMOKE_ZH.md`](experiments/blur_aware_cross_dataset/BLUR_CONDITIONED_LEGS_SMOKE_ZH.md).
The exact current stack, module boundaries, rollback matrix, and three-domain
matched smoke results are also summarized in Chinese in
[`CURRENT_ARCHITECTURE_ZH.md`](experiments/blur_aware_cross_dataset/CURRENT_ARCHITECTURE_ZH.md).
Dataset paths are intentionally external; pass a local JSON file with
`--scene-config` and a released Learn2Splat checkpoint with `--checkpoint`.

Initialize and build the exact LeGS dependency with:

```bash
git submodule update --init --recursive third_party/LeGS
PYTHON_BIN=/path/to/python optgs/scripts/install_legs_fastgs.sh
```

The submodule is pinned to LeGS commit
`8eb120b1f0c0fe0727e0440f4e372b412f275572`. The build script applies one
out-of-tree CUDA 12.x header compatibility patch to a temporary copy; the
official source tree remains unchanged.

The original interactive demo is retained below.

A learned optimizer for 3D Gaussian Splatting. This Space SfM-initializes a
COLMAP scene and refines the Gaussians live in your browser: pick the
Learn2Splat optimizer (dense or sparse checkpoint) or a 3DGS Adam baseline,
press **Start**, and watch the decoder render converge — then explore the
finished splats in an interactive 3D viewer.

Runs `demo.py --with-gui gradio` from the
[Learn2Splat repository](https://github.com/autonomousvision/learn2splat);
the optimization is rendered on the GPU and streamed by gradio, and the
result loads into a `Model3D` splat viewer.

> Requires GPU hardware. The demo holds two checkpoints in VRAM at once —
> an A10G (24 GB) is recommended.
