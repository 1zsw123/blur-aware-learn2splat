#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
HERE="$REPO/experiments/blur_aware_cross_dataset"
RUNNER="$HERE/run_cross_dataset.py"
RUN_TAG="${RUN_TAG:-current_c73eb75_m4_nima06_s1}"
OUTPUT_ROOT="$BASE/outputs/learn2splat_m4_evssm_deblurnerf21_50k_${RUN_TAG}"
LOG_ROOT="$BASE/outputs/logs/learn2splat_m4_evssm_deblurnerf21_50k_${RUN_TAG}"
SCENE_CONFIG="$LOG_ROOT/scenes_deblurnerf21_frozen.json"
SEED="${SEED:-20260831}"

SCENES=(
  motion_blurcoffee defocus_cisco
  motion_blurball defocus_bush
  motion_blurbasket defocus_cake
  motion_blurbuick defocus_caps
  motion_blurdecoration defocus_coral
  motion_blurgirl defocus_cupcake
  motion_blurheron defocus_cups
  motion_blurparterre defocus_daisy
  motion_blurpuppet defocus_sausage
  motion_blurstair defocus_seal
  defocus_tools
)
GPUS=(2 3)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

"$PYTHON" - "$HERE" "$SCENE_CONFIG" <<'PY'
import json, sys
from pathlib import Path
here, output = map(Path, sys.argv[1:])
main = json.loads((here / "scenes.json").read_text())
remaining = json.loads((here / "scenes_deblurnerf_remaining18.json").read_text())
bush = json.loads((here / "scenes_deblurnerf_bush_seal_evssm.json").read_text())
merged = dict(remaining)
merged["motion_blurcoffee"] = main["motion_blurcoffee"]
merged["defocus_cisco"] = main["defocus_cisco"]
merged["defocus_bush"] = bush["defocus_bush"]
expected = {
    "motion_blurball", "motion_blurbasket", "motion_blurbuick",
    "motion_blurcoffee", "motion_blurdecoration", "motion_blurgirl",
    "motion_blurheron", "motion_blurparterre", "motion_blurpuppet",
    "motion_blurstair", "defocus_bush", "defocus_cake", "defocus_caps",
    "defocus_cisco", "defocus_coral", "defocus_cupcake", "defocus_cups",
    "defocus_daisy", "defocus_sausage", "defocus_seal", "defocus_tools",
}
if set(merged) != expected or len(merged) != 21:
    raise RuntimeError(f"scene-set mismatch: {sorted(set(merged) ^ expected)}")
for scene, cfg in merged.items():
    for key in ("data_dir", "raw_dir", "evssm_dir", "depth_dir", "sharp_json"):
        path = Path(cfg[key])
        if not path.exists():
            raise FileNotFoundError(f"{scene}: missing {key}: {path}")
    if "turtle" in cfg["evssm_dir"].lower():
        raise RuntimeError(f"{scene}: Turtle target entered EVSSM experiment")
output.write_text(json.dumps(merged, indent=2) + "\n")
PY
sha256sum "$SCENE_CONFIG" > "$LOG_ROOT/scene_config.sha256"

for scene in "${SCENES[@]}"; do
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  }
done

cat > "$LOG_ROOT/contract.tsv" <<EOF
method	M4 full Blur-LeGS
target	EVSSM
sharp_policy	frozen per-scene NIMA>0.6 / w10 contract
scenes	21 (Motion10 + Defocus11)
steps	50000
eval_steps	10000,20000,30000,40000,50000
lpips	AlexNet v0.1 enabled
seed	$SEED
EOF

declare -A PID_GPU=()
declare -A PID_SCENE=()
PIDS=()
NEXT=0
FAILED=0

launch_scene() {
  local scene="$1" gpu="$2" log="$LOG_ROOT/${scene}.log"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$SCENE_CONFIG" \
    --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 50000 --eval-steps 10000,20000,30000,40000,50000 \
    --seed "$SEED" --objective blur-aware --optimizer learned_projected \
    --adc legs_blur --decoder-backend fastgs --densification-reward off \
    --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --coupled-dual-bpn --legs-local-objective \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    > "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_SCENE["$pid"]="$scene"
  NEXT=$((NEXT + 1))
}

for gpu in "${GPUS[@]}"; do
  launch_scene "${SCENES[$NEXT]}" "$gpu"
done

while ((${#PIDS[@]})); do
  DONE=""
  set +e
  wait -n -p DONE "${PIDS[@]}"
  STATUS=$?
  set -e
  [[ -n "$DONE" ]] || exit 3
  GPU="${PID_GPU[$DONE]}"
  SCENE="${PID_SCENE[$DONE]}"
  KEEP=()
  for pid in "${PIDS[@]}"; do [[ "$pid" == "$DONE" ]] || KEEP+=("$pid"); done
  PIDS=("${KEEP[@]}")
  if ((STATUS == 0)); then
    printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
      >> "$LOG_ROOT/status.tsv"
    if ((FAILED == 0 && NEXT < ${#SCENES[@]})); then
      launch_scene "${SCENES[$NEXT]}" "$GPU"
    fi
  else
    printf '%s\tFAILED(%s)\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$STATUS" \
      "$SCENE" "$GPU" >> "$LOG_ROOT/status.tsv"
    FAILED=1
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  fi
done

printf '%s\tQUEUE_TERMINAL\tlaunched=%s\tfailed=%s\n' \
  "$(date --iso-8601=seconds)" "$NEXT" "$FAILED" >> "$LOG_ROOT/status.tsv"
((FAILED == 0 && NEXT == ${#SCENES[@]}))
