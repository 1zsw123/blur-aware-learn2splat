# PRISM3D 50K Longitudinal Check and Laplacian Ablation

This receipt freezes the two-scene 50K continuation of the accepted
Learn2Splat/Blur-LeGS dilation-2 pipeline and a matched `sunflowers`
Laplacian-weight ablation. All reported metrics use the existing official hold
protocol. RAW blurred frames are used as the left-hand input in the qualitative
comparisons.

## Accepted pipeline results

| Scene | Step | PSNR | SSIM | LPIPS | Primitives |
|---|---:|---:|---:|---:|---:|
| jars2 | 10K | 35.3841 | 0.8930 | 0.0996 | 280,900 |
| jars2 | 20K | 36.7235 | 0.9116 | 0.0631 | 360,567 |
| jars2 | 30K | 36.8694 | 0.9138 | 0.0583 | 360,567 |
| jars2 | 40K | **37.1166** | **0.9161** | 0.0558 | 360,567 |
| jars2 | 50K | 37.0822 | 0.9160 | **0.0545** | 360,567 |
| sunflowers | 10K | 37.7800 | 0.9372 | 0.0772 | 216,721 |
| sunflowers | 20K | 39.2811 | 0.9490 | 0.0593 | 270,007 |
| sunflowers | 30K | 39.9883 | 0.9527 | 0.0547 | 270,007 |
| sunflowers | 40K | 40.4460 | 0.9548 | 0.0523 | 270,007 |
| sunflowers | 50K | **40.6213** | **0.9558** | **0.0515** | 270,007 |

Aggregate hold metrics continue to improve. The fixed strong-blur
`sunflowers` views do not: their output-reference PSNR decreases from
`34.1601/32.2281/32.2516/31.4419` at 10K to
`31.8954/31.2484/30.7900/30.5611` at 50K. This is a real longitudinal
failure hidden by the aggregate hold average and is preserved in the uploaded
RAW-input comparison.

## Matched Laplacian-weight ablation

Only `laplacian_loss_weight` changes from `0.1` to `0.2`; the scene, seed,
50K budget, renderer, BPN, dilation, capacity controller, teacher, and
evaluation protocol remain matched.

| Variant at 50K | PSNR | SSIM | LPIPS | Primitives |
|---|---:|---:|---:|---:|
| accepted weight 0.1 | **40.6213** | **0.9558** | **0.0515** | 270,007 |
| weight 0.2 | 40.3251 | 0.9536 | 0.0542 | **231,787** |

The stronger Laplacian term reduces capacity by 14.2% but regresses every
official image-quality metric. On the same strong-blur views it also fails to
recover the 10K output. It is therefore a negative ablation and is not the new
default.

## Reproduction and artifact boundary

- Accepted 50K launcher: `run_prism3d_dilation2_fix_50k_scene_s1.sh`.
- Interrupted `jars2` recovery: `run_prism3d_jars2_dilation2_fix_50k_recovery_s2.sh`.
- Matched Laplacian ablation: `run_prism3d_sunflowers_lap02_50k_s2.sh`.
- Fixed-view visualization: `combine_raw_10k_50k_visual.py`.
- Intermediate-state visualization: `make_intermediate_variable_visuals.py`
  produces matched RAW/teacher/Ours panels, colored projections of the latent
  Gaussian centers, and measured capacity/action/reward curves. The accepted
  PRISM3D run uses Turtle step-24K as its teacher; an EVSSM panel is labeled as
  a comparison baseline rather than an intermediate of that run.
- The published `point_cloud.ply` and `blur_aware_objective.pt` files are
  inference/render checkpoints. They do not include optimizer state and must
  not be described as resumable training checkpoints.
