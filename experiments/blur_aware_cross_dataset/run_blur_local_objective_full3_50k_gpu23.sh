#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUN_TAG="${RUN_TAG:-s1}"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_full3_50k_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_full3_50k_${RUN_TAG}"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
SCENES=(motion_blurcoffee defocus_cisco tum_fr2_xyz)
GPUS=(2 3)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in "${SCENES[@]}"; do
  if [[ -e "$OUTPUT_ROOT/$scene/blur-aware" ]]; then
    echo "refusing to overwrite existing output: $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  fi
done

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

declare -A PID_GPU=()
declare -A PID_SCENE=()
PIDS=()
NEXT_SCENE=0
FAILED=0

launch_scene() {
  local scene="$1"
  local gpu="$2"
  local log="$LOG_ROOT/${scene}.log"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --steps 50000 \
    --eval-steps 10000,20000,30000,40000,50000 \
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
    --skip-lpips \
    > "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_SCENE["$pid"]="$scene"
  NEXT_SCENE=$((NEXT_SCENE + 1))
}

launch_scene "${SCENES[0]}" "${GPUS[0]}"
launch_scene "${SCENES[1]}" "${GPUS[1]}"

while ((${#PIDS[@]} > 0)); do
  DONE_PID=""
  set +e
  wait -n -p DONE_PID "${PIDS[@]}"
  STATUS=$?
  set -e
  [[ -n "$DONE_PID" ]] || exit 3
  GPU="${PID_GPU[$DONE_PID]}"
  SCENE="${PID_SCENE[$DONE_PID]}"
  REMAINING=()
  for pid in "${PIDS[@]}"; do
    [[ "$pid" == "$DONE_PID" ]] || REMAINING+=("$pid")
  done
  PIDS=("${REMAINING[@]}")
  if ((STATUS == 0)); then
    printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$SCENE" "$GPU" \
      >> "$LOG_ROOT/status.tsv"
    if ((FAILED == 0 && NEXT_SCENE < ${#SCENES[@]})); then
      launch_scene "${SCENES[$NEXT_SCENE]}" "$GPU"
    fi
  else
    printf '%s\tFAILED(%s)\t%s\tGPU%s\n' \
      "$(date --iso-8601=seconds)" "$STATUS" "$SCENE" "$GPU" \
      >> "$LOG_ROOT/status.tsv"
    FAILED=1
  fi
done

((NEXT_SCENE == ${#SCENES[@]})) || exit 4
exit "$FAILED"
