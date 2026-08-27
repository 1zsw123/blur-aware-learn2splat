#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
SCENE_CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_deblurnerf_remaining18.json"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_deblurnerf_remaining18_50k_local_joint_s2"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_deblurnerf_remaining18_50k_local_joint_s2"

cd "$REPO"

for scene in defocus_seal defocus_coral; do
  if [[ -e "$OUTPUT_ROOT/$scene/blur-aware/receipt.json" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\tGPU3_REBALANCED\n' \
      "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
    continue
  fi
  if [[ -e "$OUTPUT_ROOT/$scene/blur-aware" ]]; then
    printf '%s\tFAILED_EXISTING_OUTPUT\t%s\tGPU3_REBALANCED\n' \
      "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
    exit 2
  fi

  while true; do
    IFS=',' read -r free util < <(
      nvidia-smi -i 3 --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    ((free >= 30000 && util <= 10)) && break
    sleep 30
  done

  printf '%s\tSTART\t%s\tGPU3_REBALANCED\n' \
    "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --scene-config "$SCENE_CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --steps 50000 \
    --eval-steps 10000,20000,30000,40000,50000 \
    --seed 20260824 \
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
    --legs-local-objective \
    --coupled-dual-bpn \
    --skip-lpips \
    > "$LOG_ROOT/${scene}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU3_REBALANCED\n' \
    "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
done
