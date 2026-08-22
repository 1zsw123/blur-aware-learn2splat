#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUN_TAG="${RUN_TAG:-s1}"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_smoke3_3k_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_smoke3_3k_${RUN_TAG}"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
GPU="${GPU:-2}"

SCENES=(motion_blurcoffee defocus_cisco tum_fr2_xyz)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in "${SCENES[@]}"; do
  if [[ -e "$OUTPUT_ROOT/$scene/blur-aware" ]]; then
    echo "refusing to overwrite existing output: $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  fi
done

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

for scene in "${SCENES[@]}"; do
  log="$LOG_ROOT/${scene}.log"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$GPU" \
    >> "$LOG_ROOT/status.tsv"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --steps 3000 \
    --eval-steps 1000,2000,3000 \
    --seed 20260822 \
    --objective blur-aware \
    --optimizer learned_projected \
    --adc legs_blur \
    --decoder-backend fastgs \
    --densification-reward off \
    --laplacian-loss-mode surplus \
    --laplacian-loss-weight 0.1 \
    --skip-lpips \
    > "$log" 2>&1
  status=$?
  set -e

  if ((status != 0)); then
    printf '%s\tFAILED(%s)\t%s\tGPU%s\n' \
      "$(date --iso-8601=seconds)" "$status" "$scene" "$GPU" \
      >> "$LOG_ROOT/status.tsv"
    exit "$status"
  fi

  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
done

