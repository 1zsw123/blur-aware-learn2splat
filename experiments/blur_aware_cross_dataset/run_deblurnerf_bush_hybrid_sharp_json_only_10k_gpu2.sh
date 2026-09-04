#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
SOURCE_CONFIG="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf_sharp_split_smoke3_10k_s2_20260831/scenes.json"
TAG="${TAG:-s4_20260831}"
OUTPUT_ROOT="$BASE/outputs/learn2splat_m4_evssm_deblurnerf_bush_hybrid_sharp_json_only_10k_$TAG"
LOG_ROOT="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf_bush_hybrid_sharp_json_only_10k_$TAG"
CONFIG="$LOG_ROOT/scenes.json"
GPU=2

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"
"$PYTHON" - "$SOURCE_CONFIG" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
source, output = map(Path, sys.argv[1:])
cfg = json.loads(source.read_text())["defocus_bush_hybrid"]
cfg["sharp_supervision_policy"] = "sharp_json_only"
cfg["evaluation_direct_supervision"] = False
output.write_text(json.dumps({"defocus_bush_hybrid_corrected": cfg}, indent=2) + "\n")
PY
sha256sum "$SOURCE_CONFIG" "$CONFIG" > "$LOG_ROOT/input_sha256.txt"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

while pgrep -f -- '--scene motion_blurcoffee_hybrid .*sharp_split_coffee_10k_s3_20260831' \
  >/dev/null; do
  sleep 30
done
sleep 5
if [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then
  echo "GPU${GPU} is not free after coffee hybrid; failing closed" >&2
  exit 3
fi

KEY=defocus_bush_hybrid_corrected
[[ ! -e "$OUTPUT_ROOT/$KEY/blur-aware" ]] || exit 2
printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$KEY" "$GPU" \
  >> "$LOG_ROOT/status.tsv"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUNNER" \
  --scene "$KEY" --scene-config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" --device cuda:0 \
  --steps 10000 --eval-steps 10000 --seed 20260831 \
  --objective blur-aware --optimizer learned_projected \
  --adc legs_blur --decoder-backend fastgs --densification-reward off \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
  --coupled-dual-bpn --legs-local-objective \
  --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
  --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
  > "$LOG_ROOT/${KEY}.log" 2>&1
printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$KEY" "$GPU" \
  >> "$LOG_ROOT/status.tsv"
