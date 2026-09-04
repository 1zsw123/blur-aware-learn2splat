#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
HERE="$REPO/experiments/blur_aware_cross_dataset"
RUNNER="$HERE/run_cross_dataset.py"
PREP="$BASE/outputs/logs/deblurnerf21_sharp_split_comparison_s3"
SOURCE_CONFIG="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf21_50k_current_c73eb75_m4_nima06_s1/scenes_deblurnerf21_frozen.json"
TAG="${TAG:-s1_20260831}"
OUTPUT_ROOT="$BASE/outputs/learn2splat_m4_evssm_deblurnerf_sharp_split_smoke3_10k_$TAG"
LOG_ROOT="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf_sharp_split_smoke3_10k_$TAG"
CONFIG="$LOG_ROOT/scenes.json"
SEED="${SEED:-20260831}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
sha256sum "$SOURCE_CONFIG" "$PREP/selection_report.json" \
  "$PREP/comparison_nima06.csv" "$PREP/PREPROCESSING_STATUS.json" \
  > "$LOG_ROOT/input_sha256.txt"

"$PYTHON" - "$SOURCE_CONFIG" "$PREP/selection_report.json" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path

source_path, report_path, output_path = map(Path, sys.argv[1:])
source = json.loads(source_path.read_text())
reports = json.loads(report_path.read_text())
selected = ("motion_blurcoffee", "defocus_bush", "motion_blurball")
generated = {}
for scene in selected:
    base = source[scene]
    old = dict(base)
    old["sharp_anchor_protocol"] = "fixed_nima_gt_0p6"
    generated[f"{scene}_nima06"] = old

    hybrid = dict(base)
    report = reports[scene]
    if report["nima_gmm"]["status"] == "PASS":
        hybrid["sharp_json"] = report["nima_gmm_selected_path"]
        decision = "scene_nima_two_component_gmm"
    else:
        decision = "fail_closed_fallback_fixed_nima_gt_0p6"
    hybrid["sharp_anchor_protocol"] = decision
    hybrid["sharp_anchor_report"] = str(report_path)
    generated[f"{scene}_hybrid"] = hybrid

output_path.write_text(json.dumps(generated, indent=2) + "\n")
PY
sha256sum "$CONFIG" > "$LOG_ROOT/scene_config.sha256"

for scene in motion_blurcoffee defocus_bush motion_blurball; do
  for arm in nima06 hybrid; do
    [[ ! -e "$OUTPUT_ROOT/${scene}_${arm}/blur-aware" ]] || {
      echo "refusing to overwrite $OUTPUT_ROOT/${scene}_${arm}/blur-aware" >&2
      exit 2
    }
  done
done

cat > "$LOG_ROOT/contract.tsv" <<EOF
comparison	fixed NIMA>0.6 versus identifiable-GMM/fail-closed-NIMA0.6 hybrid
scenes	motion_blurcoffee,defocus_bush,motion_blurball
pairing	same scene, same GPU, same seed, baseline then hybrid
evaluation_manifest	unchanged per scene
steps	10000
seed	$SEED
EOF

run_arm() {
  local scene="$1"
  local arm="$2"
  local gpu="$3"
  local key="${scene}_${arm}"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$key" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
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
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$key" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
}

worker() {
  local scene="$1" gpu="$2"
  run_arm "$scene" nima06 "$gpu"
  run_arm "$scene" hybrid "$gpu"
}

worker motion_blurcoffee 1 & PID1=$!
worker defocus_bush 2 & PID2=$!
worker motion_blurball 3 & PID3=$!
STATUS1=0; STATUS2=0; STATUS3=0
wait "$PID1" || STATUS1=$?
wait "$PID2" || STATUS2=$?
wait "$PID3" || STATUS3=$?
printf '%s\tQUEUE_TERMINAL\tGPU1=%s\tGPU2=%s\tGPU3=%s\n' \
  "$(date --iso-8601=seconds)" "$STATUS1" "$STATUS2" "$STATUS3" \
  >> "$LOG_ROOT/status.tsv"
((STATUS1 == 0 && STATUS2 == 0 && STATUS3 == 0))
