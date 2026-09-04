#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_exblur_bench_nima06.generated.json"
OUTPUT_ROOT="$ROOT/outputs/prism3d_exblur_bench_kernel_turtle_ablation10k_s1"
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
)

for path in \
  "$OUTPUT_ROOT/prism3d_exblur_bench/blur-aware" \
  "$OUTPUT_ROOT/prism3d_exblur_bench_turtle/blur-aware"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to reuse existing output: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
    --scene prism3d_exblur_bench \
    --bpn-kernel-size 13 \
    --bpn-kernel-dilation 2 \
    "${COMMON[@]}"
) >"$OUTPUT_ROOT/logs/evssm_kernel13.log" 2>&1 &
kernel_pid=$!

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
    --scene prism3d_exblur_bench_turtle \
    --bpn-kernel-size 9 \
    --bpn-kernel-dilation 2 \
    "${COMMON[@]}"
) >"$OUTPUT_ROOT/logs/turtle_kernel9.log" 2>&1 &
turtle_pid=$!

echo "evssm_kernel13_pid=$kernel_pid"
echo "turtle_kernel9_pid=$turtle_pid"

status=0
wait "$kernel_pid" || status=$?
wait "$turtle_pid" || status=$?
exit "$status"
