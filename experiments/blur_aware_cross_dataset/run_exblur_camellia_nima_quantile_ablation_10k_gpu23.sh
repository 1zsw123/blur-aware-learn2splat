#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUNNER="$HERE/run_cross_dataset.py"
CONFIG="$BASE/outputs/logs/exblur_camellia_nima_quantile_ablation_s1/scenes.json"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_evssm_exblur_camellia_nima_quantile_10k_s1"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_evssm_exblur_camellia_nima_quantile_10k_s1"
FULL_ROOT="$BASE/outputs/learn2splat_legs_blur_evssm_exblur8_holdtrain_nimaw10_50k_s2"

GPU2_SCENES=(exblur_camellia_nimatop25 exblur_camellia_nimatop15)
GPU3_SCENES=(exblur_camellia_nimatop20 exblur_camellia_nimatop10)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in "${GPU2_SCENES[@]}" "${GPU3_SCENES[@]}"; do
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  }
done

"$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

configs = json.load(open(sys.argv[1]))
if len(configs) != 4:
    raise SystemExit(f"expected four NIMA quantile arms, found {len(configs)}")
for scene, cfg in configs.items():
    assert cfg["sharp_supervision_policy"] == "sharp_json_only", scene
    assert cfg["evaluation_direct_supervision"] is False, scene
    assert cfg["exclude_evaluation_from_optimization"] is False, scene
    assert cfg["require_sharp_evaluation_targets"] is False, scene
    assert cfg["hold_blind_training"] is True, scene
print("HOLD_BLIND_NIMA_QUANTILE_GATE=PASS")
PY

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

run_scene() {
  local scene="$1" gpu="$2"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 10000 --eval-steps 10000 \
    --seed 20260824 \
    --objective blur-aware \
    --optimizer learned_projected \
    --adc legs_blur \
    --decoder-backend fastgs \
    --densification-reward off \
    --laplacian-loss-mode surplus \
    --laplacian-loss-weight 0.1 \
    --legs-blur-quality-weight 1.0 \
    --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 \
    --legs-blur-ramp-iters 3000 \
    --legs-local-objective \
    --coupled-dual-bpn \
    > "$LOG_ROOT/${scene}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

worker() {
  local gpu="$1"; shift
  local scene
  for scene in "$@"; do run_scene "$scene" "$gpu"; done
}

wait_for_full_gpu3() {
  while pgrep -af run_cross_dataset.py \
    | grep -F "$FULL_ROOT" \
    | grep -F -- "--scene exblur_sunflowers " >/dev/null; do
    sleep 60
  done
}

worker 2 "${GPU2_SCENES[@]}" & PID2=$!
(
  wait_for_full_gpu3
  worker 3 "${GPU3_SCENES[@]}"
) & PID3=$!
STATUS2=0; STATUS3=0
wait "$PID2" || STATUS2=$?
wait "$PID3" || STATUS3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$STATUS2" "$STATUS3" >> "$LOG_ROOT/status.tsv"
((STATUS2 == 0 && STATUS3 == 0))
