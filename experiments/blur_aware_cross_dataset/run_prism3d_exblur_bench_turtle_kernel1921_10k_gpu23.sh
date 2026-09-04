#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_exblur_bench_nima06.generated.json"
COMMON=(
  --scene prism3d_exblur_bench_turtle
  --scene-config "$CONFIG"
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
  --bpn-kernel-dilation 2
)

run_kernel() {
  local gpu=$1
  local kernel=$2
  local output_root="$ROOT/outputs/prism3d_exblur_bench_turtle_kernel${kernel}_10k_s1"
  local scene_root="$output_root/prism3d_exblur_bench_turtle/blur-aware"

  if [[ -e "$scene_root" ]]; then
    echo "Refusing to reuse existing output: $scene_root" >&2
    return 2
  fi

  mkdir -p "$output_root/logs"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
    --output-root "$output_root" \
    --bpn-kernel-size "$kernel" \
    "${COMMON[@]}" \
    >"$output_root/logs/turtle_kernel${kernel}.log" 2>&1
}

run_kernel 2 19 &
pid19=$!
run_kernel 3 21 &
pid21=$!

echo "kernel19_pid=$pid19"
echo "kernel21_pid=$pid21"

status=0
wait "$pid19" || status=$?
wait "$pid21" || status=$?
exit "$status"
