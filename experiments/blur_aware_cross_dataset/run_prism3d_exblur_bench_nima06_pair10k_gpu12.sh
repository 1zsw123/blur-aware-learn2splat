#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_exblur_bench_nima06.generated.json"
OUTPUT_ROOT="$BASE/outputs/prism3d_exblur_bench_nima06_pair10k_s1"
LOG_ROOT="$BASE/outputs/logs/prism3d_exblur_bench_nima06_pair10k_s1"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

run_scene() {
  local scene="$1" gpu="$2"
  local output="$OUTPUT_ROOT/$scene/blur-aware"
  [[ ! -e "$output" ]] || {
    echo "refusing to overwrite $output" >&2
    return 2
  }
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 10000 --eval-steps 10000 --seed 20260824 \
    --objective blur-aware --optimizer learned_projected \
    --adc legs_blur --decoder-backend fastgs \
    --densification-reward off \
    --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    --legs-local-objective --coupled-dual-bpn \
    > "$LOG_ROOT/$scene.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

run_scene control_exblur_bench 2 & control_pid=$!
run_scene prism3d_exblur_bench 1 & prism_pid=$!
control_status=0
prism_status=0
wait "$control_pid" || control_status=$?
wait "$prism_pid" || prism_status=$?
printf '%s\tTERMINAL\tcontrol=%s\tprism3d=%s\n' \
  "$(date --iso-8601=seconds)" "$control_status" "$prism_status" >> "$LOG_ROOT/status.tsv"
((control_status == 0 && prism_status == 0))
