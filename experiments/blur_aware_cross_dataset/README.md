# Cross-Dataset Blur-Aware Learn2Splat

This experiment replaces the former Triangle Splatting optimizer with the
official dense Learn2Splat checkpoint. Motion, defocus, and TUM use one
optimization contract. Dataset entries contain paths, calibrated camera
transforms, and benchmark splits only; no dataset name enters the optimizer or
capacity policy.

## Method contract

1. Fuse all trusted SfM points with available sensor/sparse depth, up to the
   checkpoint's declared evaluation initialization count. This is 70K for the
   released dense checkpoint; it is distinct from the controller's online
   active-capacity budget. Missing depth on a calibrated view is allowed;
   synthetic depth is never invented. Depth quota is distributed only among
   views with valid measurements, so an RGB-only auxiliary camera cannot
   dilute an otherwise identical initializer. Depth is converted from its declared
   measurement convention (`z` or ray `range`) before backprojection.
   Downsampling also respects the measurement topology: if every valid depth
   sample can fit in the output lattice, samples are forward-projected and
   collisions retain the nearer surface; oversampled dense depth uses nearest
   resampling. This preserves isolated COLMAP depth without introducing a
   dataset label or an empirical sparsity threshold.
2. Optimize the Gaussian representation with the official Learn2Splat learned
   optimizer and supervision-balanced FPS minibatches. If there are `N_s`
   sharp and `N_b` other input views, cumulative weighted fair scheduling uses

   ```text
   pi_sharp = 10 N_s / (10 N_s + N_b)
   ```

   of the view exposures for sharp observations. FPS within both pools retains
   camera-space coverage. This implements the same w10 risk for short and long
   reconstruction sequences; multiplying a loss after a rare frame happens to
   be sampled does not. Evaluation-reference availability is resolved
   separately from this training mask: a camera having authoritative RAW
   ground truth does not by itself make that input NIMA-sharp. Distinct cameras
   are always selected first; only when a quality pool contains fewer cameras
   than its batch quota is an observation repeated to realize the requested
   supervision mass exactly.
3. Supervise known sharp views directly. For other views, combine the EVSSM
   target with a RAW image-formation term using a representation-independent
   reliability score. The sampler realizes NIMA-sharp w10, so the loss does not
   apply it a second time. Legacy FPS/random samplers retain normalized
   per-batch w10 for controlled ablations. The validated default adds signed,
   antialiased Laplacian matching at scales 1, 2, and 4 only on trusted
   non-sharp EVSSM targets:

   ```text
   L_lap = (1 - sharp) confidence
           sum_d alpha_d rho(P_d(Render) - P_d(EVSSM))
   L_total = L_reconstruction + 0.1 L_lap + L_BPN_regularization
   ```

   `--laplacian-loss-mode surplus` is a newer ablation that treats EVSSM as a
   confidence-gated edge floor. Per-view EMA/variance and scene consensus lower
   EVSSM confidence only after stable render-over-teacher sharpness surplus;
   confidence-weighted headroom permits stronger aligned edges, while an
   unsupported-edge term penalizes noise. Strict Motion/Defocus smoke tests
   improved average SSIM/LPIPS over exact matching but did not improve average
   PSNR, so `spatial` remains the default. `--laplacian-loss-weight 0` is the
   exact pre-Laplacian rollback. Full equations and receipts are in
   `LAPLACIAN_ABLATION.md`.
4. Use a factorized BPN: one positive normalized kernel per camera and a shared
   low-resolution blur mask. It cannot store scene texture in per-pixel kernels.
5. Adapt capacity from normalized residual support and occupied capacity:

   ```text
   p_i = g_i / sum_j g_j
   S = 1 / (N_visible * sum_i p_i^2)
   C_base = C_active / visibility
   C_eff = min(C_hard, max(N, C_base / (1 - S + eps)))
   rho = N / C_eff
   d = S * (1 - rho)
   growth = clamp(d / (d + rho + eps), 0, 0.5)
   ```

   The capacity equation follows from setting unresolved residual support equal
   to fractional capacity headroom, `S = (C_eff - C_base) / C_eff`. All terms
   are dimensionless. Thus the checkpoint's 100K active set remains an
   architecture prior rather than a cross-dataset hard cap: broad unresolved
   structure can raise the demand cap, while concentrated residuals converge
   back toward the reference budget.

   The experimental `--densification-reward surplus_probe` mode closes the
   loop between this structural action and image quality. Immediately before
   each structural event, the same fixed optimization-only probe views measure
   confidence-weighted target PSNR `Q_t` and supported render-over-EVSSM
   Laplacian surplus `U_t`. For the action taken during the preceding interval:

   ```text
   dQ_t = Q_t - Q_(t-1)                 sQ_t = EMA(|dQ_t|)
   dU_t = U_t - U_(t-1)                 sU_t = EMA(|dU_t|)
   q_t  = mean[tanh(dQ_t / sQ_t), tanh(dU_t / sU_t)]
   c_t  = previous_growth_budget / previous_effective_cap
   r_t  = q_t - c_t                     rbar_t = EMA(r_t)
   a_t  = 2 sigmoid(rbar_t)
   ```

   The surplus component is omitted only when no valid non-sharp probe exists.
   `a_t=1` is neutral; it multiplies residual support and only the excess part
   of the inferred demand cap. This is a delayed densification-action reward,
   not a loss on held-out frames. Its normalization uses online absolute
   increments rather than a hand-set PSNR/surplus threshold, so a smaller but
   still positive convergence gain is not misclassified as failure.
   `--densification-reward probe_control` renders the identical fixed probes
   but does not feed them to ADC, providing an instrumentation-matched control.
6. For mixed-camera COLMAP scenes, apply each camera's calibrated crop/resize
   transform and require one shared output shape. Virtual FrameCrafter views are
   train-only and are rejected if they overlap the explicit evaluation split.
7. Derive the optimization budget from the released checkpoint's expected
   supervision-risk exposure rather than a dataset label. Let `m_i` be the
   sampler mass (10 for known sharp, 1 otherwise):

   ```text
   N_eff = sum_i m_i / max_i m_i
   T = max(T_checkpoint,
           ceil(T_checkpoint * N_eff / N_checkpoint_context_views))
   ```

   The official 42-view TUM mapping set therefore keeps the checkpoint's 2K
   learned-proposal budget. The nonbenchmark 3,397-view full-stream diagnostic
   would receive 14,132 learned updates. Bounded smoke tests use this rule;
   paper evaluations use the same 50K total horizon for every dataset, with
   the objective-consistent projection stage starting after the checkpoint's
   2K learned horizon.
8. Restart only Learn2Splat's recurrent latent and gradient-normalizer state at
   its released 2K training horizon. The current Gaussians, BPN, sampler state,
   and accumulated capacity diagnostics remain continuous. The released Dense
   checkpoint disables explicit time encoding, so this is a recurrent-state
   distribution safeguard, not a claim about `t/2000`. Checkpoints that enable
   time encoding also receive a local-time reset. The reset is evaluated
   against an otherwise identical continuous-rollout ablation.
9. For runs longer than the released learned horizon, the cross-dataset final
   solver uses Learn2Splat as a 2K learned proposal and then applies an
   objective-consistent Adam residual projection. The handoff is determined by
   checkpoint metadata, not a dataset label. BPN, supervision sampling, and the
   scale-free capacity rule remain unchanged, so the second stage converges the
   same scene objective instead of replacing it with a TUM-specific loss.

## Scene-Adaptive Sharp Anchor Discovery

The hold-blind pipeline does not use a global NIMA score or a manually selected
top percentage. It robustly normalizes all NIMA scores within a scene, fits a
two-component Gaussian mixture, and assigns frames to the higher-quality
component at posterior probability greater than `0.5`. The selected ratio is
therefore scene-dependent. Selected anchors receive direct RAW supervision,
BPN bypass, and the existing `w10` weighted-fair sampling; benchmark hold/test
identities remain evaluation-only metadata.

On ExBlur, the method automatically selects 39/244 frames versus 38/244 frozen
hold frames. It recovers all 38 hold frames and adds only
`stone_lantern/029`, giving 100% recall and 97.44% precision/Jaccard overlap.
This selection diagnostic does not substitute for the ongoing matched 50K
reconstruction evaluation. Equations, fail-closed gates, protocol boundaries,
the per-scene table, and reproduction commands are in
[`SCENE_ADAPTIVE_SHARP_ANCHORS.md`](SCENE_ADAPTIVE_SHARP_ANCHORS.md).

## Evaluation protocol

- Deblur-NeRF Motion/Defocus: optimize all input frames and evaluate every name
  in the frozen NIMA>0.6 sharp manifest. The Motion manifest equals its legacy
  hold subset; Defocus uses the complete manifest rather than an arithmetic
  subset. Legacy `hold=N` remains a fail-closed fallback only when no explicit
  benchmark manifest is configured.
- I2-SLAM/Unblur-SLAM TUM `fr2_xyz`: the released system consumes the complete
  associated 32 Hz sequence for tracking, but `MotionFilter` inserts only the
  42 indices in `scripts/fr2_xyz_indices.txt` into its mapping video. We
  therefore optimize those exact 42 mapping keyframes and evaluate their
  authoritative RAW observations. Training sharp supervision is independently
  frozen by the NIMA>0.6 manifest (currently 1/42); the other 41 observations
  retain EVSSM/BPN/Laplacian supervision. The indices are positions in the
  filtered stream, not raw TUM filenames.
- Metrics are computed against RAW sharp observations, never EVSSM targets or
  generated views. Final runs report PSNR, SSIM, and LPIPS-AlexNet v0.1.

The 3,397-view full-stream optimization and FrameCrafter scenes are retained
only as diagnostic ablations. Neither is substituted for the released
Unblur-SLAM/I2-SLAM mapping protocol. The final controlled FrameCrafter pair
optimizes 42 observed plus five generated train-only views and evaluates the
same authoritative 42 RAW frames. Generated views do not receive sharp w10;
they use independent confidence weights and cannot dilute depth
initialization. Surplus improves the corrected augmented pair by only 0.0804
dB at 10K with mixed SSIM/LPIPS behavior, and the best augmented arm remains
0.1902 dB below the unaugmented baseline. Augmentation is therefore not a
default method component. See `LAPLACIAN_ABLATION.md` for both the failed w10
pseudo-sharp diagnostic and the corrected pair, and
`UPSTREAM_PROTOCOL_AUDIT.md` for the code-level protocol audit.

## Densification-action reward smoke

The strict test uses one scene from each domain, 10K updates, two seeds
(`20260821`, `20260822`), `learned_projected`, support-conditioned `adaptive`
ADC, surplus loss weight `0.1`, and identical fixed-probe rendering in both
arms. Only ADC consumption of the delayed action reward changes. Bars in the
generated figure are two-seed means and dots are individual runs.

| Domain / scene | Control PSNR | Reward PSNR | Delta PSNR | Delta SSIM | Delta LPIPS | Delta primitives |
|---|---:|---:|---:|---:|---:|---:|
| Motion / blurcoffee | 41.5858 | **41.7026** | **+0.1168** | **+0.000402** | **-0.003502** | +1,607 |
| Defocus / cisco | 34.0323 | **34.0488** | **+0.0165** | **+0.000529** | +0.000275 | +6,350 |
| TUM / fr2_xyz | 25.5405 | **25.6289** | **+0.0884** | **+0.002864** | +0.001404 | +7,060 |

Across all six paired runs, mean deltas are `+0.0739 dB` PSNR,
`+0.001265` SSIM, `-0.000608` LPIPS, and `+5,005` primitives. SSIM improves in
all six runs; PSNR improves in four of six and has a positive mean in every
domain. LPIPS improves consistently only on Motion, while Defocus and TUM
regress slightly. CUDA raster/gradient aggregation is not bitwise deterministic
and the two arms can diverge before the first reward is consumed, so tiny
single-run deltas are not treated as causal proof. This is a positive
cross-domain mechanism smoke, not a claim that reward dominates every metric
or a substitute for multi-scene 50K evaluation. TUM renders also retain visible
geometry artifacts despite their metric gain.

The hash-bound CSV/JSON, capacity-event audit, and plot are under
`outputs/learn2splat_surplus_action_reward_cross_dataset_s2_final`.

## Exact LeGS transplantation

`--adc legs --decoder-backend fastgs` is the newest capacity candidate and a
separately controlled reference ablation. It keeps Learn2Splat as the Gaussian
parameter optimizer but transplants the
released LeGS structural controller rather than approximating it with the
global adaptive policy:

- official LeGS commit `8eb120b1f0c0fe0727e0440f4e372b412f275572`;
- official FastGS CUDA leave-one-out L1 sensitivity and alpha visibility;
- ten randomly sampled training cameras per decision;
- normalized 11-D state: XYZ gradients (3), scale gradients (3), opacity
  gradient (1), DC-color gradients (3), and sensitivity (1);
- per-Gaussian keep/clone/split actor and separate low-opacity prune estimator;
- parent-child aggregation after the released 50-step reward delay;
- two-transition GAE/PPO update, two epochs, 500K chunks, and the released
  learning-rate schedule;
- released 500/100/15000 structural schedule, 3K opacity reset, no global
  primitive cap, and 15K-post final opacity pruning.

The representation update, BPN/Laplacian objective, data protocol, and final
hold metrics remain this repository's Learn2Splat pipeline. Consequently this
is an exact transplantation of the LeGS *capacity mechanism*, not a claim that
the whole LeGS training stack is reproduced. A fair comparison must run both
controllers with `--decoder-backend fastgs`; comparing LeGS/FastGS against the
default gsplat renderer would mix controller and renderer effects. The former
short-horizon rescaled/global policy is an adapted ablation and is not exposed
as `--adc legs`.

```bash
git submodule update --init --recursive third_party/LeGS
PYTHON_BIN="$ENV" optgs/scripts/install_legs_fastgs.sh

CUDA_VISIBLE_DEVICES=2 "$ENV" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene motion_blurcoffee --output-root "$OUT/exact_legs" \
  --steps 10000 --eval-steps 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
  --objective blur-aware --optimizer learned_projected --adc legs \
  --decoder-backend fastgs --densification-reward off \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1
```

### Verified three-domain matched smoke

Exact LeGS was compared with the current adaptive-surplus controller on one
representative Motion, Defocus, and TUM scene. Every pair uses the same
Learn2Splat optimizer, FastGS decoder, seed `20260822`, data, objective, hold
frames, initialization, and 1K--10K metric schedule. Only the capacity
controller and its native reward path differ.

| Domain / scene | Adaptive best PSNR / SSIM | Exact LeGS best PSNR / SSIM | Best PSNR delta | 10K PSNR delta | 10K Gaussian ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Motion / blurcoffee | 39.754 / 0.9827 @8K | **46.003 / 0.9934 @9K** | **+6.250** | **+6.357** | 6.37x |
| Defocus / cisco | 32.798 / 0.9602 @10K | **34.595 / 0.9700 @9K** | **+1.797** | **+1.485** | 6.39x |
| TUM / fr2_xyz | 25.116 / 0.8347 @10K | **26.634 / 0.8689 @9K** | **+1.518** | **+1.265** | 8.76x |

The Defocus render is visibly cleaner with no large black/red structural
artifact. TUM is sharper but retains minor white speckles consistent with its
8.8x capacity increase. All exact runs peak at 9K and regress slightly at
10K. These are positive single-scene, single-seed, 10K mechanism smokes;
LPIPS was skipped for turnaround, and they are not a multi-scene 50K
generalization or efficiency claim. The hash-bound summary is generated by
`summarize_adc_pairs.py`. A Chinese end-to-end architecture and limitation
summary is in `CURRENT_ARCHITECTURE_ZH.md`.

## Reproduction

```bash
ENV=/path/to/learn2splat-python
OUT=/path/to/learn2splat-cross-dataset-outputs

CUDA_VISIBLE_DEVICES=2 $ENV experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene motion_blurcoffee --output-root "$OUT" --device cuda:0 \
  --steps 50000 --eval-steps 10000,20000,30000,40000,50000 \
  --adc adaptive --optimizer learned_projected \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
  --densification-reward surplus_probe

CUDA_VISIBLE_DEVICES=3 $ENV experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene defocus_cisco --output-root "$OUT" --device cuda:0 \
  --steps 50000 --eval-steps 10000,20000,30000,40000,50000 \
  --adc adaptive --optimizer learned_projected \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
  --densification-reward surplus_probe

CUDA_VISIBLE_DEVICES=2 $ENV experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene tum_fr2_xyz --output-root "$OUT" --device cuda:0 \
  --steps 50000 --eval-steps 10000,20000,30000,40000,50000 \
  --adc adaptive --optimizer learned_projected \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
  --densification-reward surplus_probe
```

All three commands use the same 50K horizon, optimizer, supervision-FPS
sampler, objective, and controller. Scene configuration supplies only paths,
calibrated camera/depth conventions, and benchmark splits. Use `--adc none`
for the no-controller ablation, `--adc adaptive` for the support-conditioned
v2 ablation, and add `--laplacian-loss-mode surplus`,
`--laplacian-loss-weight 0.1`, and
`--densification-reward surplus_probe` for the action-reward arm. Replace only
the last value with `probe_control` for its fixed-probe-matched control. Omit
`--steps` only for the
checkpoint-exposure smoke protocol. The old Triangle Splatting repository and
outputs are not modified by this branch, so rollback means running the
existing Tri launcher rather than changing this experiment.
