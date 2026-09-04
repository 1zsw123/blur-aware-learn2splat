#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
OUTPUT_ROOT="$ROOT/outputs/prism3d_e8_current_dilation2_fix_10k_s1"
LOG_ROOT="$OUTPUT_ROOT/logs"
STATUS="$LOG_ROOT/status.tsv"

COMMON=(
  --scene-config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
  --steps 10000
  --eval-steps 10000
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
  --latent-blur-assignment
  --opt-batch-strategy fps
)

SCENES=(
  prism3d_bench_turtle
  prism3d_dragon_turtle
  prism3d_jars_turtle
  prism3d_jars2_turtle
  prism3d_postbox_turtle
  prism3d_sunflowers_turtle
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
printf 'timestamp\tstate\tscene\tgpu\n' > "$STATUS"

run_scene() {
  local gpu=$1
  local scene=$2
  printf '%s\tSTART\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$STATUS"
  if (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      experiments/blur_aware_cross_dataset/run_cross_dataset.py \
      --scene "$scene" "${COMMON[@]}"
  ) > "$LOG_ROOT/${scene}.log" 2>&1; then
    printf '%s\tCOMPLETE\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$STATUS"
  else
    local rc=$?
    printf '%s\tFAILED:%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$rc" "$scene" "$gpu" >> "$STATUS"
    return "$rc"
  fi
}

worker() {
  local gpu=$1
  shift
  local scene
  for scene in "$@"; do
    run_scene "$gpu" "$scene"
  done
}

worker 1 prism3d_bench_turtle prism3d_jars2_turtle &
pid1=$!
worker 2 prism3d_dragon_turtle prism3d_postbox_turtle &
pid2=$!
worker 3 prism3d_jars_turtle prism3d_sunflowers_turtle &
pid3=$!

status=0
wait "$pid1" || status=$?
wait "$pid2" || status=$?
wait "$pid3" || status=$?
exit "$status"
