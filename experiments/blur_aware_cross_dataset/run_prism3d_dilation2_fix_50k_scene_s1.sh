#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 PHYSICAL_GPU {jars2|sunflowers}" >&2
  exit 2
fi

GPU=$1
SHORT_SCENE=$2
case "$SHORT_SCENE" in
  jars2|sunflowers) ;;
  *) echo "unsupported scene: $SHORT_SCENE" >&2; exit 2 ;;
esac

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
PYTHON="$ROOT/envs/learn2splat-py312/bin/python"
CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_prism3d_e8_turtle_step024000_nima06_identityblind.generated.json"
OUTPUT_ROOT="$ROOT/outputs/prism3d_jars2_sunflowers_current_dilation2_fix_50k_s1"
LOG_ROOT="$OUTPUT_ROOT/logs"
SCENE="prism3d_${SHORT_SCENE}_turtle"

[[ -x "$PYTHON" ]]
[[ -f "$CONFIG" ]]
[[ ! -e "$OUTPUT_ROOT/$SCENE/blur-aware" ]] || {
  echo "Refusing to overwrite existing output: $OUTPUT_ROOT/$SCENE/blur-aware" >&2
  exit 2
}
mkdir -p "$LOG_ROOT"
printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
  >> "$LOG_ROOT/status.tsv"

if (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiments/blur_aware_cross_dataset/run_cross_dataset.py \
    --scene "$SCENE" \
    --scene-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --steps 50000 \
    --eval-steps 10000,20000,30000,40000,50000 \
    --seed 20260902 \
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
    --latent-blur-assignment \
    --opt-batch-strategy fps
) > "$LOG_ROOT/${SCENE}.log" 2>&1; then
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
else
  rc=$?
  printf '%s\tFAILED:%s\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$rc" "$SCENE" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
  exit "$rc"
fi
