#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
OUTPUT_ROOT="$ROOT/outputs/prism3d_jars_pair_identity_artifact_capacityfix_20k_s1"
LOG_ROOT="$OUTPUT_ROOT/logs"

COMMON=(
  --scene-config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
  --steps 20000
  --eval-steps 10000,20000
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
  --bpn-kernel-dilation 1
  --latent-blur-assignment
  --opt-batch-strategy fps
)

SCENES=(
  prism3d_jars_turtle
  prism3d_jars2_turtle
)

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
  local gpu=$1
  local scene=$2
  echo "[$(date --iso-8601=seconds)] START scene=$scene gpu=$gpu"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      experiments/blur_aware_cross_dataset/run_cross_dataset.py \
      --scene "$scene" "${COMMON[@]}"
  ) >"$LOG_ROOT/${scene}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] COMPLETE scene=$scene gpu=$gpu"
}

run_scene 2 prism3d_jars_turtle &
pid_jars=$!
run_scene 3 prism3d_jars2_turtle &
pid_jars2=$!

status=0
wait "$pid_jars" || status=$?
wait "$pid_jars2" || status=$?
exit "$status"
