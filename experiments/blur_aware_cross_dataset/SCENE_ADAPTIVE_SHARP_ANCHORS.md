# Scene-Adaptive Sharp Anchor Discovery

## Motivation

A fixed NIMA threshold is not calibrated across scenes, while a fixed top
percentage assumes every sequence contains the same fraction of useful sharp
observations. Both assumptions fail when blur prevalence and score scale change
between Deblur-NeRF, ExBlur, and robot sequences. The method therefore estimates
the sharp-anchor population independently inside each scene. It never receives
benchmark hold/test identities.

## Algorithm

For scene NIMA scores `s_i`, first apply robust scene normalization:

```text
z_i = (s_i - median(s)) / MAD(s)
```

Fit a deterministic two-component Gaussian mixture to all `z_i` values. The
component with the larger mean is the candidate sharp population. Frame `i` is
selected when

```text
P(high-quality component | z_i) > 0.5.
```

Consequently, the scene ratio

```text
p_scene = number of selected frames / number of scene frames
```

is an output rather than a tuned hyperparameter. The posterior boundary is the
equal-cost Bayesian decision boundary, not a dataset-specific NIMA threshold.
The implementation uses a fixed random seed and multiple GMM initializations.
It fails closed when the scores are constant, the two-component BIC does not
improve over one component, the standardized component separation is below 2,
or the posterior assignment is not a monotonic upper-tail split. These are
identifiability checks, not a selected-frame ratio.

## Training contract

Only the selected frame-name JSON enters training. Selected anchors receive:

- direct RAW supervision;
- BPN bypass, because the observation is treated as relatively sharp;
- the existing cumulative weighted-fair `w10` sampling mass.

All other observations retain EVSSM/BPN/Laplacian supervision. With
`hold_blind_training=true`, policy probes are selected from the optimization
inputs without consulting the evaluation set. `evaluation_direct_supervision`
and `require_sharp_evaluation_targets` must both be false. Benchmark hold names
are loaded only by the post-training evaluator, whose targets are always RAW.

This is an offline/transductive reconstruction protocol when evaluation camera
observations are part of the optimized sequence. It must not be described as
novel-view holdout training. A strict novel-view protocol can instead set
`exclude_evaluation_from_optimization=true`, but that is a different benchmark
contract.

## ExBlur sanity check

The selector was fitted to all input-frame NIMA scores in each of the eight
ExBlur scenes. Official hold identities were read only after fitting for this
diagnostic comparison.

| Scene | Frames | Official hold | Automatic anchors | Hold overlap |
|---|---:|---:|---:|---:|
| bench | 36 | 6 (16.67%) | 6 (16.67%) | 6/6 |
| camellia | 34 | 5 (14.71%) | 5 (14.71%) | 5/5 |
| dragon | 23 | 3 (13.04%) | 3 (13.04%) | 3/3 |
| jars | 24 | 4 (16.67%) | 4 (16.67%) | 4/4 |
| jars2 | 26 | 6 (23.08%) | 6 (23.08%) | 6/6 |
| postbox | 25 | 5 (20.00%) | 5 (20.00%) | 5/5 |
| stone_lantern | 33 | 4 (12.12%) | 5 (15.15%) | 4/4 |
| sunflowers | 43 | 5 (11.63%) | 5 (11.63%) | 5/5 |
| **Total** | **244** | **38 (15.57%)** | **39 (15.98%)** | **38/38** |

The sole additional automatic anchor is `stone_lantern/029`. Aggregate hold
recall is 100%, precision is 97.44%, and Jaccard overlap is 97.44%. This confirms
that ExBlur's designated hold observations form a visually sharp population,
but does not by itself establish reconstruction quality. The current controlled
run evaluates the extra `029` anchor first, then runs the remaining seven
scenes under the identical label-free selection contract.

## Reproduction

Discover anchors for one scene without any split file:

```bash
python experiments/blur_aware_cross_dataset/scene_adaptive_sharp_anchors.py \
  --scores /path/to/scene_nima_koniq_scores.json \
  --anchors /path/to/scene_nima_gmm_sharp_frames.json \
  --report /path/to/scene_nima_gmm_report.json
```

The score manifest is a JSON list with unique `name` and numeric `nima_koniq`
fields. Use the generated anchor JSON as `sharp_json` in the scene config and
set:

```json
{
  "sharp_supervision_policy": "sharp_json_only",
  "evaluation_direct_supervision": false,
  "exclude_evaluation_from_optimization": false,
  "require_sharp_evaluation_targets": false,
  "hold_blind_training": true
}
```

`analyze_exblur_nima_adaptive_ratio.py` is an evaluation-only utility that
compares already-discovered anchors with the frozen ExBlur hold lists. It is not
used to produce training anchors.

For all eight ExBlur scenes, generate hold-blind anchors and scene configs
directly from score manifests with:

```bash
python experiments/blur_aware_cross_dataset/prepare_exblur_nima_adaptive_gmm.py \
  --base-config /path/to/exblur_scenes.json \
  --scores-root /path/to/nima_score_manifests \
  --output-dir /path/to/generated_hold_blind_config
```

This generator has no hold/test-manifest argument. Its `selection_report.json`
contains selection diagnostics only, not benchmark overlap statistics.
