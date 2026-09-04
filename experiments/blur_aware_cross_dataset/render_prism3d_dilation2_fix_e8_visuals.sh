#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
PAIR_ROOT="$ROOT/outputs/prism3d_camellia_stone_kernel25_dilation2_10k_s1"
SIX_ROOT="$ROOT/outputs/prism3d_e8_current_dilation2_fix_10k_s1"
RETRY_ROOT="$ROOT/outputs/prism3d_bench_jars2_current_dilation2_fix_10k_s2"

render_scene() {
  local gpu=$1
  local root=$2
  local scene=$3
  local output="$root/prism3d_${scene}_turtle/blur-aware/blurred_input_vs_output_top3.png"
  [[ ! -e "$output" ]] || return 0
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      experiments/blur_aware_cross_dataset/render_blurred_input_output.py \
      --scene "prism3d_${scene}_turtle" \
      --run-root "$root" \
      --scene-config "$CONFIG" \
      --output "$output" \
      --device cuda:0 \
      --rows 3
  )
}

worker() {
  local gpu=$1
  shift
  while (( $# > 0 )); do
    render_scene "$gpu" "$1" "$2"
    shift 2
  done
}

worker 2 \
  "$RETRY_ROOT" bench \
  "$SIX_ROOT" dragon \
  "$SIX_ROOT" postbox \
  "$SIX_ROOT" sunflowers &
pid2=$!

worker 3 \
  "$PAIR_ROOT" camellia \
  "$SIX_ROOT" jars \
  "$RETRY_ROOT" jars2 \
  "$PAIR_ROOT" stone_lantern &
pid3=$!

status=0
wait "$pid2" || status=$?
wait "$pid3" || status=$?
exit "$status"
