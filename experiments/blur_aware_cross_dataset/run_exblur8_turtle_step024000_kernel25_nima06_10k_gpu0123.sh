#!/usr/bin/env bash
set -uo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_exblur8_turtle_step024000_nima06.generated.json"
OUTPUT_ROOT="$ROOT/outputs/exblur8_turtle_step024000_kernel25_nima06_10k_s1"
COMMON=(
  --scene-config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
  --steps 10000
  --eval-steps 10000
  --seed 20260824
  --objective blur-aware
  --optimizer learned_projected
  --adc legs_blur
  --decoder-backend fastgs
  --densification-reward off
  --laplacian-loss-mode surplus
  --laplacian-loss-weight 0.1
  --legs-blur-quality-weight 1.0
  --legs-blur-capacity-weight 0.10
  --legs-blur-start-iter 2000
  --legs-blur-ramp-iters 3000
  --legs-local-objective
  --coupled-dual-bpn
  --bpn-kernel-size 25
  --bpn-kernel-dilation 2
)

mkdir -p "$OUTPUT_ROOT/logs"

run_scene() {
  local gpu=$1
  local scene=$2
  local scene_root="$OUTPUT_ROOT/$scene/blur-aware"
  if [[ -e "$scene_root" ]]; then
    echo "Refusing to reuse existing output: $scene_root" >&2
    return 2
  fi
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
    --scene "$scene" \
    "${COMMON[@]}" \
    >"$OUTPUT_ROOT/logs/${scene}.log" 2>&1
}

run_wave() {
  local -a scenes=("$@")
  local -a pids=()
  local status=0
  for gpu in 0 1 2 3; do
    run_scene "$gpu" "${scenes[$gpu]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  return "$status"
}

status=0
run_wave exblur_bench exblur_camellia exblur_dragon exblur_jars || status=$?
run_wave exblur_jars2 exblur_postbox exblur_stone_lantern exblur_sunflowers || status=$?
exit "$status"
