#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
HERE="$REPO/experiments/blur_aware_cross_dataset"
RUNNER="$HERE/run_cross_dataset.py"
TAG="${TAG:-s1_20260831}"
OUTPUT_ROOT="$BASE/outputs/learn2splat_latent_blur_assignment_smoke3_10k_$TAG"
LOG_ROOT="$BASE/outputs/logs/learn2splat_latent_blur_assignment_smoke3_10k_$TAG"
CONFIG="$LOG_ROOT/scenes.json"
EMPTY_SHARP="$LOG_ROOT/no_predeclared_sharp_frames.json"
SEED="${SEED:-20260831}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"
printf '[]\n' > "$EMPTY_SHARP"
"$PYTHON" - "$HERE/scenes.json" "$EMPTY_SHARP" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
source_path, empty_sharp, output_path = map(Path, sys.argv[1:])
source = json.loads(source_path.read_text())
mapping = {
    "motion_blurcoffee_latent": "motion_blurcoffee",
    "defocus_cisco_latent": "defocus_cisco",
    "tum_fr2_xyz_latent": "tum_fr2_xyz",
}
generated = {}
for alias, scene in mapping.items():
    cfg = dict(source[scene])
    cfg["sharp_json"] = str(empty_sharp)
    cfg["sharp_supervision_policy"] = "sharp_json_only"
    cfg["evaluation_direct_supervision"] = False
    cfg["latent_blur_assignment"] = True
    cfg["protocol_description"] = (
        "hold-blind latent blur assignment; no predeclared sharp frames; "
        "evaluation identity is retained for metrics only"
    )
    generated[alias] = cfg
output_path.write_text(json.dumps(generated, indent=2) + "\n")
PY

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
sha256sum "$HERE/scenes.json" "$EMPTY_SHARP" "$CONFIG" \
  > "$LOG_ROOT/input_sha256.txt"

for scene in motion_blurcoffee_latent defocus_cisco_latent tum_fr2_xyz_latent; do
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  }
done

cat > "$LOG_ROOT/contract.tsv" <<EOF
method	Latent Blur Assignment
sharp_labels	none
evaluation_identity	metrics only
sampling	fps (no frozen sharp sampler)
steps	10000
eval_steps	10000
seed	$SEED
EOF

run_scene() {
  local scene="$1"
  local gpu="$2"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 10000 --eval-steps 10000 --seed "$SEED" \
    --opt-batch-strategy fps \
    --objective blur-aware --optimizer learned_projected \
    --adc legs_blur --decoder-backend fastgs --densification-reward off \
    --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --coupled-dual-bpn --latent-blur-assignment --legs-local-objective \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    > "$LOG_ROOT/${scene}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
}

worker2() {
  run_scene motion_blurcoffee_latent 2
  run_scene tum_fr2_xyz_latent 2
}

worker2 & PID2=$!
run_scene defocus_cisco_latent 3 & PID3=$!
STATUS2=0; STATUS3=0
wait "$PID2" || STATUS2=$?
wait "$PID3" || STATUS3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' \
  "$(date --iso-8601=seconds)" "$STATUS2" "$STATUS3" >> "$LOG_ROOT/status.tsv"
((STATUS2 == 0 && STATUS3 == 0))
