#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
HERE="$BASE/repos/learn2splat-official-space/experiments/blur_aware_cross_dataset"
EVSSM_REPO="$BASE/repos/EVSSM"
EVSSM_PYTHON="/home/szha0669/miniconda3/envs/evssm/bin/python"
CHECKPOINT="$BASE/checkpoints/evssm_unblurslam/net_g_latest.pth"
STAGE="$BASE/data/exblurnerf_evssm_stage"
OUTPUT="$BASE/data/evssm_deblurred_exblurnerf"
LOG="$BASE/outputs/logs/evssm_exblur8_prepare_s1"

GPU2_SCENES=(bench dragon jars2 stone_lantern)
GPU3_SCENES=(camellia jars postbox sunflowers)
mkdir -p "$OUTPUT" "$LOG"

worker() {
  local gpu="$1"; shift
  local scene
  for scene in "$@"; do
    mkdir -p "$OUTPUT/$scene"
    CUDA_VISIBLE_DEVICES="$gpu" "$EVSSM_PYTHON" "$BASE/scripts/evssm_fast_infer.py" \
      --data_dir "$STAGE/$scene" \
      --test_model "$CHECKPOINT" \
      --result_dir "$OUTPUT/$scene" \
      --batch_size 1 --num_workers 2 \
      > "$LOG/${scene}.log" 2>&1
  done
}

(cd "$EVSSM_REPO" && worker 2 "${GPU2_SCENES[@]}") & PID2=$!
(cd "$EVSSM_REPO" && worker 3 "${GPU3_SCENES[@]}") & PID3=$!
wait "$PID2"
wait "$PID3"

for scene in "${GPU2_SCENES[@]}" "${GPU3_SCENES[@]}"; do
  expected=$(find "$STAGE/$scene/test/input" -maxdepth 1 -type l | wc -l)
  actual=$(find "$OUTPUT/$scene" -maxdepth 1 -type f -name '*.png' | wc -l)
  [[ "$actual" -eq "$expected" ]] || {
    echo "$scene: EVSSM output count $actual != $expected" >&2
    exit 3
  }
done
