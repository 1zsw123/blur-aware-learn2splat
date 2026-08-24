#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_shared_residual_refine_smoke3_10k_s1"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_shared_residual_refine_smoke3_10k_s1"
FULL3_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_full3_50k_local_joint_s1"
TUM3_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_tum3_full50k_local_joint_s2"

SCENES=(motion_blurcoffee defocus_cisco tum_fr2_xyz)
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in "${SCENES[@]}"; do
  if [[ "$scene" == tum_fr2_xyz ]]; then
    source_root="$TUM3_ROOT"
  else
    source_root="$FULL3_ROOT"
  fi
  source_dir="$source_root/$scene/blur-aware"
  test -s "$source_dir/point_cloud.ply"
  test -s "$source_dir/blur_aware_objective.pt"
  test ! -e "$OUTPUT_ROOT/$scene/blur-aware"

  printf '%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$scene" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --output-root "$OUTPUT_ROOT" \
    --initial-ply "$source_dir/point_cloud.ply" \
    --initial-objective-state "$source_dir/blur_aware_objective.pt" \
    --device cuda:0 \
    --steps 10000 \
    --eval-steps 5000,10000 \
    --seed 20260824 \
    --objective blur-aware \
    --optimizer adam \
    --adc none \
    --decoder-backend fastgs \
    --densification-reward off \
    --laplacian-loss-mode surplus \
    --laplacian-loss-weight 0.1 \
    --coupled-dual-bpn \
    --skip-lpips \
    > "$LOG_ROOT/$scene.log" 2>&1
  printf '%s\tCOMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" \
    >> "$LOG_ROOT/status.tsv"
done
