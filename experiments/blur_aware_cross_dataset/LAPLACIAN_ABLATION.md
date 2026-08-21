# Spatial Laplacian Objective

## Question

Can edge supervision improve deblurring without rewarding arbitrary image
noise or duplicating direct supervision on known-sharp frames?

## Failed global-energy formulation

The first formulation compressed each image into one global sharpness scalar:

```text
g_E = log Lap(E) - log Lap(B)
g_R = log Lap(R) - log Lap(B)
L_lap = relu(g_E - g_R)                       # non-sharp views
L_lap = abs(log Lap(R) - log Lap(B))          # known-sharp views
```

This loss cannot identify where an edge belongs. A sparse bright impulse can
increase `Lap(R)` enough to compensate for a missing edge elsewhere. It also
duplicates RGB/SSIM supervision on known-sharp frames. The 0.01 Motion run
visibly exhibited this failure as sparse bright artifacts.

### Motion Blurcoffee, global energy

| Weight | 5K PSNR | 5K LPIPS | 10K PSNR | 10K SSIM | 10K LPIPS |
|---:|---:|---:|---:|---:|---:|
| 0 | 35.8762 | 0.09458 | 40.5152 | 0.98425 | 0.03768 |
| 0.001 | **36.3557** | **0.08120** | **40.5176** | **0.98493** | **0.03661** |
| 0.003 | 36.0703 | 0.08553 | 40.1235 | 0.98326 | 0.04567 |
| 0.01 | 35.8456 | 0.08150 | 39.5757 | 0.98414 | 0.04465 |

The `0.001` arm helps early convergence and LPIPS, but larger weights create
sparse high-frequency bright artifacts and reduce final PSNR.

### Defocus Cisco, global energy

| Weight | 5K PSNR | 5K LPIPS | 10K PSNR | 10K SSIM | 10K LPIPS |
|---:|---:|---:|---:|---:|---:|
| 0 | **31.8606** | **0.04701** | **33.7542** | **0.96472** | **0.03814** |
| 0.001 | 31.6872 | 0.04762 | 33.5480 | 0.96399 | 0.03903 |

The smallest Motion-positive weight does not transfer to Defocus. It loses
0.2062 dB at 10K and worsens both SSIM and LPIPS.

## Corrected spatial formulation

For RAW input `B_i`, EVSSM target `E_i`, render `R_i`, static EVSSM
confidence `c_i`, and known-sharp indicator `s_i`, define the signed,
antialiased Laplacian response at scale `d` as:

```text
P_d(I) = Laplacian(Gaussian(grayscale(downsample(I, d))))
d in {1, 2, 4}
alpha = normalize([1, 0.5, 0.25])
rho(x) = sqrt(x^2 + 1e-6) - 1e-3

L_lap(i) = (1 - s_i) c_i sum_d alpha_d mean_valid rho(P_d(R_i) - P_d(E_i))
L_total  = L_reconstruction + 0.1 L_lap + L_BPN_regularization
```

The signed spatial response retains edge location and polarity. Gaussian
prefiltering suppresses one-pixel noise, the three scales cover fine and broad
structure, and Charbonnier distance prevents a few outliers from dominating.
Known-sharp frames receive no Laplacian term because their direct target is
already the authoritative RAW image. An all-sharp batch short-circuits the
entire Laplacian computation.

The comparison below uses identical seeds, data, evaluation frames, 10K
schedules, `adaptive_legacy`, and `learned_projected` settings.

| Scene | Mode / weight | 10K PSNR | 10K SSIM | 10K LPIPS | Delta PSNR | Delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Motion blurcoffee | off / 0 | 40.5152 | 0.98425 | 0.03768 | - | - |
| Motion blurcoffee | spatial / 0.1 | **41.0847** | **0.98494** | **0.03400** | **+0.5695** | **-0.00367** |
| Defocus cisco | off / 0 | 33.7542 | 0.96472 | 0.03814 | - | - |
| Defocus cisco | spatial / 0.1 | **33.7738** | **0.96481** | **0.03792** | **+0.0196** | **-0.00022** |

## Dynamic EVSSM-surplus formulation

Exact spatial matching still treats EVSSM as the edge-amplitude ceiling. The
experimental `surplus` mode instead uses EVSSM as a confidence-gated edge
floor and permits a render to become sharper when that surplus is spatially
supported and stable across training views.

Let `q_d(I)` be the evidence-supported Laplacian energy at scale `d`, and let
`g_E` and `g_R` be the multiscale log-energy gains over RAW:

```text
g_E = sum_d alpha_d [log q_d(E) - log q_d(B)]
g_R = sum_d alpha_d [log q_d(R) - log q_d(B)]
delta_i = g_R - g_E
```

For every non-sharp training view, an EMA tracks `delta_i` and its square.
Only the variance-discounted positive surplus contributes to scene consensus:

```text
mu_i, nu_i <- EMA(delta_i), EMA(delta_i^2)
p_i = relu(mu_i - sqrt(relu(nu_i - mu_i^2))) * (1 - beta^n_i)
h = tanh(mean_nonsharp p_i)
u_i = tanh(relu(mu_i - sigma_i)) * (1 - beta^n_i) * h
c_eff_i = c_i (1 - u_i)
```

The maturity factor is continuous and threshold-free. An instantaneous sharp
sample cannot lower its own EVSSM confidence. The desired signed edge
amplitude is

```text
a_i = exp(0.5 * ((1 - c_i) tanh(relu(g_E)) + h))
```

where `0.5` converts log energy into log amplitude. At each scale, the loss
combines an under-sharp floor `D`, a soft reliable-teacher overshoot anchor
`O`, and unsupported high-frequency artifact penalty `A`:

```text
D = relu(a_i |P(E)| - sign(P(E)) P(R))
O = relu(sign(P(E)) P(R) - a_i |P(E)|)
A = relu(|P(R)| - a_i max(|P(E)|, |P(B)|))

L_surplus(i) = (1 - s_i) [
    c_eff_i rho(D)
  + c_eff_i^2 support rho(O)
  + (1 - support) rho(A)
]
```

Thus stronger aligned edges are possible, but flat-region noise cannot create
teacher uncertainty. Known-sharp views short-circuit the complete branch.

### Strict 10K smoke result

All rows use the same seed, data, hold frames, `adaptive_legacy`, and
`learned_projected`. `surplus-v2` is the implementation above.

| Scene | Mode | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Motion blurcoffee | off | 40.5152 | 0.98425 | 0.03768 |
| Motion blurcoffee | exact spatial | **41.0847** | 0.98494 | 0.03400 |
| Motion blurcoffee | surplus-v2 | 40.9289 | **0.98603** | **0.03183** |
| Defocus cisco | off | 33.7542 | 0.96472 | 0.03814 |
| Defocus cisco | exact spatial | **33.7738** | **0.96481** | **0.03792** |
| Defocus cisco | surplus-v2 | 33.5388 | 0.96404 | 0.03863 |
| Two-scene average | off | 37.1347 | 0.97448 | 0.03791 |
| Two-scene average | exact spatial | **37.4292** | 0.97488 | 0.03596 |
| Two-scene average | surplus-v2 | 37.2338 | **0.97503** | **0.03523** |

The dynamic formulation improves Motion perceptual quality and the two-scene
average over Laplacian-off, but it does not dominate exact spatial matching in
PSNR and regresses Defocus. A follow-up that increased the overshoot anchor
from the scene's sharp-frame fraction recovered Defocus PSNR (33.7418) but
reduced Motion PSNR (40.5091); it was rejected as a hidden dataset-mixture
tuning direction.

The first TUM `fr2_xyz` smoke incorrectly promoted all 42 evaluation cameras
to training-time known-sharp views because each camera had an authoritative
RAW metric reference. That made both Laplacian-off and surplus report exactly
zero Laplacian loss. Its 25.8188 and 25.9450 dB results are retained only as an
invalid protocol diagnostic; their difference cannot be attributed to the
objective. The corrected protocol keeps all 42 RAW references for evaluation
but derives the training sharp mask independently from the NIMA>0.6 manifest
(1 sharp, 41 EVSSM/BPN views).

The first corrected-mask run then exposed a second edge case: batch-wise
without-replacement sampling capped the single sharp view at 12.5% exposure,
below the w10 target of 19.6078%. The sampler now repeats an observation only
when a quality pool is smaller than its requested batch quota. Unit tests cover
both the ordinary no-duplicate case and this one-sharp limit. The final paired
TUM smoke realizes 19.6075% exposure in both arms:

| TUM `fr2_xyz`, official 42 RAW references | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Laplacian off | **25.5237** | **0.83140** | **0.11023** |
| Surplus, weight 0.1 | 25.4264 | 0.83002 | 0.11122 |
| Surplus minus off | -0.0973 | -0.00138 | +0.00100 |

The surplus branch is active (`1` sharp, `41` non-sharp), but does not improve
TUM at 10K. EVSSM already measures 37.47 dB against RAW on the 41 non-sharp
views, while the reconstruction remains near 25.5 dB. During the final 1K
updates, mean render sharpness gain is -0.115 versus +0.0056 for EVSSM, so the
representation has not reached the teacher's fidelity and its dynamic
render-over-teacher uncertainty remains near zero. Applying the edge floor at
that stage slightly competes with geometry/photometric convergence instead of
adding useful multiview surplus.

### FrameCrafter train-only augmentation

A second strict pair adds five FrameCrafter views to the 42 released TUM
mapping keyframes. Both arms optimize all 47 views and evaluate the same 42
authoritative RAW observations as the unaugmented protocol. Generated views
never enter evaluation. The seed, optimizer, controller, schedule, and sampler
are identical; only the Laplacian weight changes.

| TUM `fr2_xyz`, official 42 RAW references | 5K PSNR | 10K PSNR | 10K SSIM | 10K LPIPS |
|---|---:|---:|---:|---:|
| No generated views, Laplacian off | 24.6465 | **25.5237** | **0.83140** | **0.11023** |
| Five generated views, Laplacian off | 22.6134 | 23.2976 | 0.78241 | 0.16189 |
| Five generated views, surplus 0.1 | **22.8901** | **23.8565** | **0.78699** | **0.15880** |
| Surplus minus augmented off | +0.2767 | **+0.5588** | **+0.00458** | **-0.00309** |

Within the augmented pair, surplus improves PSNR on 32/42 views and LPIPS on
31/42 views. The gain is therefore larger than run noise and answers the
controlled question positively: Laplacian supervision helps after adding the
generated views under that old weighting protocol.

The augmentation itself is not successful. Even the augmented surplus arm is
1.6672 dB below the unaugmented Laplacian-off run. The five generated views
were labelled pseudo-sharp together with the one NIMA-sharp observed view, so
w10 raises their combined sampling exposure to 59.406%. Their hidden observed
frame checks average only 16.0088 dB (range 14.2531--17.7631 dB), despite
passing sharpness and DINO-semantic gates, and all five lack depth
initialization. A sharp-looking, semantically consistent generated image is
therefore not a sufficiently accurate multiview photometric target. These
frames must not receive authoritative-sharp w10 supervision in the paper
pipeline. A follow-up should use confidence-weighted train-only supervision
and require geometric/photometric consistency, while retaining the same 42
frame evaluation protocol.

The follow-up removes that confound. Only the one observed NIMA-sharp view
receives w10. The five generated views bypass BPN as direct pseudo targets but
remain outside the sharp pool and receive independent confidence weights
between 0.1624 and 0.1938. Their confidence uses only train-time overlap,
depth confidence, DINO consistency, and sharpness-ratio agreement; hidden RAW
targets are never read. Missing-depth auxiliary cameras also no longer consume
the depth initialization quota, restoring the exact unaugmented initialization
of 50,000 SfM plus 19,835 depth points in both arms.

| TUM `fr2_xyz`, official 42 RAW references | 5K PSNR | 10K PSNR | 10K SSIM | 10K LPIPS |
|---|---:|---:|---:|---:|
| No generated views, Laplacian off | **24.6465** | **25.5237** | **0.83140** | **0.11023** |
| Confidence-weighted generated views, Laplacian off | 24.4358 | 25.2531 | **0.82892** | 0.11448 |
| Confidence-weighted generated views, surplus 0.1 | 24.4646 | **25.3335** | 0.82757 | **0.11348** |
| Surplus minus corrected augmented off | +0.0288 | +0.0804 | -0.00135 | -0.00100 |

In this fair pair, surplus improves PSNR on only 16/42 views and LPIPS on
25/42 views. It gives a small aggregate PSNR/LPIPS improvement but lowers
SSIM, so it is a weak mixed effect rather than a robust augmentation gain. The
best corrected augmented arm remains 0.1902 dB below the unaugmented run, with
SSIM lower by 0.00383 and LPIPS higher by 0.00325. Visual grids contain no
black regions or pose catastrophes; the remaining gap is consistent with the
generated targets' limited photometric accuracy. FrameCrafter augmentation is
therefore retained as a negative diagnostic and is not part of the default
pipeline.

## Decision

The corrected spatial objective with weight 0.1 passes the representative
Motion and Defocus cross-domain PSNR gate and remains the default. Dynamic
surplus mode is retained as a reproducible perceptual-quality ablation, not
promoted as a universal improvement. The failed global energy formulation
remains available as `--laplacian-loss-mode energy` only for reproducibility.
`--laplacian-loss-weight 0` is the exact rollback.

This is a representative-scene mechanism gate, not a full-dataset claim.

Receipts and visual grids are stored in:

```text
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s72_motion_lap0_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s73_motion_lap001_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s74_motion_lap0001_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s75_motion_lap0003_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s76_defocus_lap0_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s77_defocus_lap0001_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s82_motion_lapspatial_nonsharp01_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s83_defocus_lapspatial_nonsharp01_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s90_motion_surplus_v2_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s91_defocus_surplus_v2_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s94_tum_lapoff_sameconfig_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s95_tum_surplus_v2_sameconfig_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s98_tum_protocolfix_w10_lapoff_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s99_tum_protocolfix_w10_surplus_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s100_tum_framecrafter42eval_w10_lapoff_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s101_tum_framecrafter42eval_w10_surplus_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s104_tum_framecrafter_confaux_depthquota_lapoff_strict10k
/srv2/szha0669/blur_slam_exp/outputs/learn2splat_cross_dataset_s105_tum_framecrafter_confaux_depthquota_surplus_strict10k
```
