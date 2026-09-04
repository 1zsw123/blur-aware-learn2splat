#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 KERNEL_GPU2 KERNEL_GPU3" >&2
  exit 2
fi

kernel2=$1
kernel3=$2
for kernel in "$kernel2" "$kernel3"; do
  if (( kernel < 3 || kernel % 2 == 0 )); then
    echo "kernel must be odd and >= 3: $kernel" >&2
    exit 2
  fi
done

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

run_kernel 2 "$kernel2" &
pid2=$!
run_kernel 3 "$kernel3" &
pid3=$!

echo "kernel${kernel2}_pid=$pid2"
echo "kernel${kernel3}_pid=$pid3"

status=0
wait "$pid2" || status=$?
wait "$pid3" || status=$?
exit "$status"
