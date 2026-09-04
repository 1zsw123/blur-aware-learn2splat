# PRISM3D Dilation-2 Fix: Eight-Scene 10K Result

All eight scenes use the same Learn2Splat/Blur-LeGS configuration: Turtle
step-24K teacher, NIMA>0.6 anchors, hold identity hidden, all input frames in
optimization, a coupled 25x25 BPN with dilation 2, RAW-neighborhood Laplacian
support, latent blur assignment, negative-quality birth veto, and the
quality-gated final-prune option.  Learned SE(3) exposure trajectories are off.

| Scene | Old PSNR | Current PSNR | Delta | SSIM | LPIPS | Primitives |
|---|---:|---:|---:|---:|---:|---:|
| bench | 32.903 | 37.313 | +4.410 | 0.9259 | 0.0405 | 379,219 |
| camellia | 30.187 | 36.486 | +6.298 | 0.9347 | 0.0275 | 615,891 |
| dragon | 37.065 | 40.014 | +2.949 | 0.9647 | 0.0233 | 450,552 |
| jars | 32.894 | 38.625 | +5.731 | 0.9484 | 0.0304 | 488,866 |
| jars2 | 32.607 | 36.641 | +4.034 | 0.9065 | 0.0775 | 360,193 |
| postbox | 33.168 | 36.587 | +3.419 | 0.9298 | 0.0420 | 516,129 |
| stone_lantern | 30.347 | 38.534 | +8.187 | 0.9489 | 0.0544 | 239,166 |
| sunflowers | 40.534 | 38.159 | -2.375 | 0.9373 | 0.0754 | 212,988 |
| **Average** | **33.713** | **37.795** | **+4.082** | **0.9370** | **0.0464** | **407,876** |

The old average was 0.8961 SSIM, 0.1028 LPIPS, and 720,409 primitives.  The
current result therefore changes average SSIM by +0.0409, LPIPS by -0.0565,
and primitive count by -312,533 (-43.4%).  Seven scenes improve in PSNR;
sunflowers is the sole regression and must not be hidden by the average.

At 10K, negative-quality birth vetoes execute, but the post-15K final opacity
prune cannot execute.  This run therefore validates the 10K objective,
assignment, local-state, reward, and birth-control behavior, but does not
experimentally validate the final-prune protection.

These are aggregate hold metrics and post-hoc RAW-input visualizations.  They
are an engineering comparison, not a component-wise causal ablation: dilation
2 and the four safeguards changed together relative to the old run.
