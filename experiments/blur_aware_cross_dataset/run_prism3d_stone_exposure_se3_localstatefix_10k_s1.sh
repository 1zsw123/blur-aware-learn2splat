#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
OUTPUT_ROOT="$ROOT/outputs/prism3d_stone_exposure_se3_localstatefix_10k_s4"
LOG_ROOT="$OUTPUT_ROOT/logs"
SCENE=prism3d_stone_lantern_turtle

[[ ! -e "$OUTPUT_ROOT/$SCENE/blur-aware" ]] || exit 2
mkdir -p "$LOG_ROOT"
cd "$REPO"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" \
  experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene "$SCENE" \
  --scene-config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda:0 \
  --steps 10000 \
  --eval-steps 10000 \
  --seed 20260903 \
  --objective blur-aware \
  --optimizer learned_projected \
  --adc legs_blur \
  --decoder-backend fastgs \
  --densification-reward off \
  --laplacian-loss-mode surplus \
  --laplacian-loss-weight 0.1 \
  --laplacian-support-mode raw_neighborhood \
  --legs-blur-quality-weight 1.0 \
  --legs-blur-capacity-weight 0.10 \
  --legs-blur-start-iter 2000 \
  --legs-blur-ramp-iters 3000 \
  --legs-local-objective \
  --legs-blur-negative-birth-veto \
  --legs-blur-quality-gated-final-prune \
  --coupled-dual-bpn \
  --bpn-kernel-size 25 \
  --bpn-kernel-dilation 2 \
  --bpn-kernel-bases 1 \
  --latent-blur-assignment \
  --exposure-trajectory-samples 3 \
  --exposure-trajectory-learning-rate 0.002 \
  --exposure-trajectory-max-rotation 0.05 \
  --exposure-trajectory-max-translation-fraction 0.05 \
  --exposure-trajectory-motion-prior 0.01 \
  --exposure-trajectory-center-prior 0.1 \
  --exposure-center-supervision-weight 0.5 \
  --opt-batch-strategy fps \
  >"$LOG_ROOT/$SCENE.log" 2>&1
