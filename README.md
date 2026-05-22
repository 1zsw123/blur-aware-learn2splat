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

A learned optimizer for 3D Gaussian Splatting. This Space SfM-initializes a
COLMAP scene and refines the Gaussians live in your browser: pick the
Learn2Splat optimizer (dense or sparse checkpoint) or a 3DGS Adam baseline,
press **Start**, and watch the splats converge.

Runs `demo.py --with-gui client` from the
[Learn2Splat repository](https://github.com/autonomousvision/learn2splat);
the splats are drawn by viser's in-browser WebGL renderer.

> Requires GPU hardware. The demo holds two checkpoints in VRAM at once —
> an A10G (24 GB) is recommended.
