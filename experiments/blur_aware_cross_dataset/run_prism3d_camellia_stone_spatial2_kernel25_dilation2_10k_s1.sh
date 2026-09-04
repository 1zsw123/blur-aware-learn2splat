#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/prism3d_camellia_stone_spatial2_kernel25_dilation2_10k_s3}"
LOG_ROOT="$OUTPUT_ROOT/logs"
STEPS="${STEPS:-10000}"
KERNEL_BASES="${KERNEL_BASES:-2}"
OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
TEACHER_MODEL_SELECTION="${TEACHER_MODEL_SELECTION:-1}"
SCENE="${SCENE:-}"
if [[ -n "$SCENE" ]]; then
  SCENES=("$SCENE")
else
  SCENES=(prism3d_camellia_turtle prism3d_stone_lantern_turtle)
fi

COMMON=(
  --scene-config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
  --steps "$STEPS"
  --eval-steps "$STEPS"
  --seed 20260902
  --objective blur-aware
  --optimizer learned_projected
  --adc legs_blur
  --decoder-backend fastgs
  --densification-reward off
  --laplacian-loss-mode surplus
  --laplacian-loss-weight 0.1
  --laplacian-support-mode raw_neighborhood
  --legs-blur-quality-weight 1.0
  --legs-blur-capacity-weight 0.10
  --legs-blur-start-iter 2000
  --legs-blur-ramp-iters 3000
  --legs-local-objective
  --legs-blur-negative-birth-veto
  --legs-blur-quality-gated-final-prune
  --coupled-dual-bpn
  --bpn-kernel-size 25
  --bpn-kernel-dilation 2
  --bpn-kernel-bases "$KERNEL_BASES"
  --latent-blur-assignment
  --opt-batch-strategy fps
  --opt-batch-size "$OPT_BATCH_SIZE"
)
if [[ "$TEACHER_MODEL_SELECTION" == 1 ]]; then
  COMMON+=(--teacher-blur-model-selection)
fi

[[ -x "$PYTHON" ]]
[[ -f "$CONFIG" ]]
for scene in "${SCENES[@]}"; do
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  }
done
mkdir -p "$LOG_ROOT"

run_scene() {
  local scene=$1
  echo "[$(date --iso-8601=seconds)] START scene=$scene gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  (
    cd "$REPO"
    PYTHONUNBUFFERED=1 "$PYTHON" experiments/blur_aware_cross_dataset/run_cross_dataset.py \
      --scene "$scene" "${COMMON[@]}"
  ) >"$LOG_ROOT/${scene}.log" 2>&1 &
  local child=$!
  while kill -0 "$child" 2>/dev/null; do
    echo "[$(date --iso-8601=seconds)] HEARTBEAT scene=$scene pid=$child"
    sleep 5
  done
  wait "$child"
  echo "[$(date --iso-8601=seconds)] COMPLETE scene=$scene gpu=${CUDA_VISIBLE_DEVICES:-unset}"
}

# Run sequentially so the experiment cannot trespass on GPUs currently held
# by other users. Set CUDA_VISIBLE_DEVICES to one verified-free physical GPU.
for scene in "${SCENES[@]}"; do
  run_scene "$scene"
done
