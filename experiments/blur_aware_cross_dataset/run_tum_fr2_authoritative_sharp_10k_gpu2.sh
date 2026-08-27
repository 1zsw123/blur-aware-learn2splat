#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUN_TAG="${RUN_TAG:-s1}"
SCENE="tum_fr2_xyz_authoritative_sharp"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_tum_fr2_authoritative_sharp_10k_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_tum_fr2_authoritative_sharp_10k_${RUN_TAG}"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
GPU="${GPU:-2}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

if [[ -e "$OUTPUT_ROOT/$SCENE/blur-aware" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT/$SCENE/blur-aware" >&2
  exit 2
fi

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
  > "$LOG_ROOT/status.tsv"

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUNNER" \
  --scene "$SCENE" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda:0 \
  --steps 10000 \
  --eval-steps 10000 \
  --seed 20260822 \
  --objective blur-aware \
  --optimizer learned_projected \
  --adc legs_blur \
  --decoder-backend fastgs \
  --densification-reward off \
  --laplacian-loss-mode surplus \
  --laplacian-loss-weight 0.1 \
  --legs-blur-quality-weight 1.0 \
  --legs-blur-capacity-weight 0.10 \
  --legs-blur-start-iter 2000 \
  --legs-blur-ramp-iters 3000 \
  > "$LOG_ROOT/$SCENE.log" 2>&1
status=$?
set -e

if ((status == 0)); then
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
else
  printf '%s\tFAILED(%s)\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$status" "$SCENE" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
fi
exit "$status"
