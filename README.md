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
- a confidence-gated, multiscale Laplacian surplus objective that allows
  supported render sharpness to exceed the EVSSM initialization; and
- a scale-free adaptive densification controller whose delayed reward uses
  fixed training probes, target PSNR improvement, Laplacian surplus
  improvement, and a primitive-growth cost.

The controller operates on Gaussian clone/split/prune decisions. It is a custom
adaptive ADC, not a LeGS/PPO implementation. The implementation, equations,
benchmark protocols, rollback flags, smoke-test table, and reproduction
commands are documented in
[`experiments/blur_aware_cross_dataset/README.md`](experiments/blur_aware_cross_dataset/README.md).
Dataset paths are intentionally external; pass a local JSON file with
`--scene-config` and a released Learn2Splat checkpoint with `--checkpoint`.

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
