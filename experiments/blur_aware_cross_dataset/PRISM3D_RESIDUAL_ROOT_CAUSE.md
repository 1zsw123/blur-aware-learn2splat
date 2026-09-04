# PRISM3D residual blur root-cause audit

## Scope

This audit concerns the remaining visible ghosting in `camellia` and
`stone_lantern` after the identity/artifact/capacity repair. Display evidence
must use the RAW blurred frame as input and the model render as output. Sharp
release images are used only for post-hoc QA and are never displayed as input.

## Confirmed causes

### 1. The old blur support was physically too small

The 25-tap BPN with dilation 1 covers only +/-12 pixels. Sharp exposure
sequences show endpoint displacement around 30--40 pixels in the selected bad
views, with 90th-percentile non-global residual motion around 14--33 pixels.
Changing only dilation from 1 to 2 expands support to +/-24 pixels and improved
the official 10K metrics:

| Scene | dilation 1 PSNR/SSIM/LPIPS | dilation 2 PSNR/SSIM/LPIPS | PSNR delta |
|---|---:|---:|---:|
| camellia | 34.9716 / 0.9159 / 0.0424 | 36.4856 / 0.9347 / 0.0275 | +1.5140 |
| stone_lantern | 37.2335 / 0.9431 / 0.0641 | 38.5336 / 0.9489 / 0.0544 | +1.3001 |

This is causal evidence, not a complete fix. Two of six fixed visual QA views
still regress against endpoint sharp GT because convolution cannot correct a
camera-pose mismatch.

### 2. A global convolution cannot represent the observed blur field

After subtracting a single global translation, selected exposure sequences
retain 14--33 pixels of 90th-percentile residual motion. This is consistent
with rotation/parallax and cannot be represented by one image-wide kernel.
The current PRISM3D inputs also have no sensor depth, so the BPN's depth channel
is identically zero.

### 3. The teacher reblur branch is saturated

At 10K, selected problem views have teacher blur strength approximately
0.996--1.000 and near-zero center mass. Thus the coupled BPN can almost fully
reblur the Turtle/EVSSM target before applying the teacher loss. This weakens
the direct constraint on the latent sharp render. The failure is especially
underdetermined because `camellia` has zero NIMA>0.6 sharp anchors and
`stone_lantern` has one.

The new optional teacher model-selection gate compares direct and reblurred
teacher hypotheses using normalized BIC/MDL evidence. The more expressive
reblur branch is used only when its log-error gain exceeds its active-kernel
parameter cost. The decision is per view and detached, so the BPN cannot game
the selector. The old behavior remains available by omitting
`--teacher-blur-model-selection`.

### 4. Fixed center poses are not the physical image-formation model

Official PRISM3D renders 14 virtual views along a learned Bezier exposure
trajectory and averages them during training. It also optimizes evaluation
poses because released sharp images may not coincide with the exposure
midpoint. The current Learn2Splat path uses one fixed VGGSfM pose and an image
plane convolution. This explains why aggregate metrics can improve while an
endpoint-referenced individual view remains shifted or doubled.

## Controlled negative results

The optional teacher BIC/MDL selector did not solve the residual artifacts.
Relative to the dilation-2 baseline it changed camellia by -0.198 dB and
stone_lantern by -0.030 dB. The learned teacher-reblur posterior remained
about 0.87 and 0.86 respectively, so the selector did not break the saturated
teacher branch.

A two-basis spatial BPN was also tested with every other setting held fixed.
It did not collapse to two identical kernels: the mean basis L1 distance was
about 1.65 out of 2. However, camellia's spatial basis-0 weight had only 0.012
pixelwise standard deviation, so the model effectively learned one global
mixture. The empty depth channel and local RAW/teacher residual are not enough
to infer the depth-varying exposure flow. Official 10K results were:

| Scene | global dilation 2 | two spatial bases | PSNR delta |
|---|---:|---:|---:|
| camellia | 36.4856 / 0.9347 / 0.0275 | 35.5299 / 0.9168 / 0.0421 | -0.9556 |
| stone_lantern | 38.5336 / 0.9489 / 0.0544 | 38.5718 / 0.9495 / 0.0542 | +0.0382 |

The fixed RAW-input visual audit agrees with the aggregate metrics: camellia
frame 030 remains doubled and stone_lantern frame 031 retains background
stretching. Therefore neither unconditional teacher reblur selection nor more
image-plane kernel bases should be promoted as the repair.

### Frozen-geometry pose intervention

A diagnostic froze the dilation-2 point cloud and optimized only small SE(3)
camera corrections. This used sharp references solely to test causality and is
not an admissible training or evaluation result. Optimizing the official hold
views changed mean PSNR by only +0.001 dB (camellia) and +0.022 dB
(stone_lantern), excluding a general evaluation-pose offset as the main cause.

The two displayed failure views give a more specific split:

| View | sharp-reference PSNR before | after pose-only fit | Delta |
|---|---:|---:|---:|
| camellia 030 | 22.0946 | 22.2873 | +0.1927 |
| stone_lantern 031 | 24.1079 | 25.3895 | +1.2816 |

Thus the stone_lantern artifact is materially pose-related, while camellia
cannot be repaired by moving the frozen camera: its doubled content has already
been encoded into the learned geometry/appearance. This is the expected failure
mode when exposure motion is represented by a post-render convolution rather
than by rendering and integrating a camera trajectory during optimization.

## Current experimental changes

- `--bpn-kernel-bases N` adds a rollback-safe low-rank spatial mixture. `N=1`
  is the previous global-kernel behavior.
- `--teacher-blur-model-selection` prevents unconditional teacher reblur using
  per-view normalized model evidence.
- Both are optional and preserve the old pipeline when disabled.
- Unit coverage: the full `tests/test_adaptive_adc.py` suite passes (64 tests).
- Multi-basis convolutions are fused by convolution group and the coupled RAW
  and teacher responses are derived from one base convolution by exact
  linearity. Numerical equivalence to the previous implementation was checked
  below 6e-7 absolute error.

## Learned SE(3) exposure-trajectory intervention

An opt-in physical exposure branch was implemented to test the remaining
hypothesis.  Each training view learns bounded opening and closing twists,
interpolates an SE(3) trajectory, renders the scene from the exposure samples,
and averages those samples for the RAW-image branch.  The center render remains
the sharp/teacher branch.  The implementation includes trajectory magnitude and
midpoint priors, a center-supervision anchor, and a straight-through FastGS
gradient proxy while preserving the correct moved-camera forward rendering.
It is disabled when `exposure_trajectory_samples=1`, which is the default.

Three controlled variants were run:

| Variant | camellia PSNR / SSIM / LPIPS | stone_lantern PSNR / SSIM / LPIPS |
|---|---:|---:|
| dilation-2 baseline | 36.4856 / 0.9347 / 0.0275 | 38.5336 / 0.9489 / 0.0544 |
| SE(3) s2, weak priors | 29.8304 / 0.7686 / 0.2525 | 33.7147 / 0.8962 / 0.1748 |
| SE(3) s3, moved-camera forward and strong priors | stopped after causal failure | 32.2892 / 0.8901 / 0.1870 |
| SE(3) s4, exposure excluded from frozen LeGS local state | not run | 30.6580 / 0.8702 / 0.2370 |

The s3 priors reduced stone_lantern's normalized trajectory midpoint drift from
0.2655 to 0.0366, but quality still fell by 6.2443 dB.  Excluding exposure from
the frozen LeGS local state made the result worse, not better.  On the fixed RAW
frame-031 audit, the baseline raises relative Laplacian sharpness by 55.1%,
whereas s4 changes it by -11.4%.  The deterioration is therefore visible in the
deblurred output and is not an evaluation-only effect.

This rejects jointly learning an unconstrained per-image exposure trajectory in
the current objective.  Geometry, BPN blur, and twelve trajectory variables per
view are non-identifiable from a single blurred observation.  Strong magnitude
priors prevent trajectory runaway but cannot identify the correct path; the
optimizer instead trades trajectory, geometry, and appearance against one
another.  A viable future physical model needs an independent trajectory cue or
a staged optimization that estimates motion against frozen geometry before
joint refinement.

## Admission boundary

The dilation-2, spatial-basis, and teacher-selection results above are
measured. Dilation 2 is the only accepted improvement; the spatial-basis,
teacher-selection, and learned-trajectory branches are negative ablations and
remain disabled by default. The full unit suite passes (68 tests). A full
physical correction requires independently constrained exposure trajectories,
not a jointly underdetermined per-view trajectory added to the current loss.
