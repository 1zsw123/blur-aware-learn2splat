#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUNNER="$HERE/run_cross_dataset.py"
SCENE_CONFIG="$HERE/scenes_exblur8_evssm.generated.json"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_evssm_exblur_holdtrain_nimaw10_smoke2_20k_s3"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_evssm_exblur_holdtrain_nimaw10_smoke2_20k_s3"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

run_scene() {
  local scene="$1" gpu="$2"
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$scene/blur-aware" >&2
    return 2
  }
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$SCENE_CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 20000 --eval-steps 10000,20000 \
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
    > "$LOG_ROOT/${scene}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

run_scene exblur_camellia 2 & PID2=$!
run_scene exblur_jars 3 & PID3=$!
STATUS2=0; STATUS3=0
wait "$PID2" || STATUS2=$?
wait "$PID3" || STATUS3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$STATUS2" "$STATUS3" >> "$LOG_ROOT/status.tsv"
((STATUS2 == 0 && STATUS3 == 0))
