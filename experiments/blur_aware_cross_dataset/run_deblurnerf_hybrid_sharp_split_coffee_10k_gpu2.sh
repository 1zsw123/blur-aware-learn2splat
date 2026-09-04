#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
CONFIG="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf_sharp_split_smoke3_10k_s2_20260831/scenes.json"
TAG="${TAG:-s3_20260831}"
OUTPUT_ROOT="$BASE/outputs/learn2splat_m4_evssm_deblurnerf_sharp_split_coffee_10k_$TAG"
LOG_ROOT="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf_sharp_split_coffee_10k_$TAG"
SEED="${SEED:-20260831}"
GPU=2

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"
sha256sum "$CONFIG" > "$LOG_ROOT/scene_config.sha256"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

# The predecessor is a separately managed exactly-once process.  Do not race
# it for memory or GPU state; this queue does not restart the predecessor.
while pgrep -f -- '--scene defocus_bush_hybrid .*sharp_split_smoke3_10k_s2_20260831' \
  >/dev/null; do
  sleep 30
done
sleep 5
if [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
  echo "GPU${GPU} is not free after predecessor; failing closed" >&2
  exit 3
fi

run_arm() {
  local arm="$1"
  local key="motion_blurcoffee_${arm}"
  [[ ! -e "$OUTPUT_ROOT/$key/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$key/blur-aware" >&2
    exit 2
  }
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$key" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUNNER" \
    --scene "$key" --scene-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 10000 --eval-steps 10000 --seed "$SEED" \
    --objective blur-aware --optimizer learned_projected \
    --adc legs_blur --decoder-backend fastgs --densification-reward off \
    --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --coupled-dual-bpn --legs-local-objective \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    > "$LOG_ROOT/${key}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$key" "$GPU" \
    >> "$LOG_ROOT/status.tsv"
}

run_arm nima06
run_arm hybrid
printf '%s\tQUEUE_TERMINAL\tPASS\n' "$(date --iso-8601=seconds)" \
  >> "$LOG_ROOT/status.tsv"
