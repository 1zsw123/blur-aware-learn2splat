#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
SCENE_CONFIG="$REPO/experiments/blur_aware_cross_dataset/scenes_deblurnerf_remaining18.json"
RUN_TAG="${RUN_TAG:-local_joint_s1}"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_deblurnerf_remaining17_50k_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_deblurnerf_remaining17_50k_${RUN_TAG}"

GPU2_SCENES=(
  motion_blurball
  motion_blurbuick
  motion_blurgirl
  motion_blurparterre
  motion_blurstair
  defocus_caps
  defocus_daisy
  defocus_seal
  defocus_coral
)
GPU3_SCENES=(
  motion_blurbasket
  motion_blurdecoration
  motion_blurheron
  motion_blurpuppet
  defocus_cake
  defocus_cups
  defocus_sausage
  defocus_tools
)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in "${GPU2_SCENES[@]}" "${GPU3_SCENES[@]}"; do
  if [[ -e "$OUTPUT_ROOT/$scene/blur-aware" ]]; then
    echo "refusing to overwrite existing output: $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  fi
done

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
printf '%s\tBLOCKED_MISSING_EVSSM_043\tdefocus_cupcake\tNONE\n' \
  "$(date --iso-8601=seconds)" >> "$LOG_ROOT/status.tsv"

wait_for_gpu() {
  local gpu="$1"
  local consecutive=0
  while ((consecutive < 2)); do
    local free util
    IFS=',' read -r free util < <(
      nvidia-smi -i "$gpu" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if ((free >= 30000 && util <= 10)); then
      consecutive=$((consecutive + 1))
    else
      consecutive=0
    fi
    printf '%s\tWAIT_GPU\tGPU%s\tfree_mb=%s util=%s stable=%s/2\n' \
      "$(date --iso-8601=seconds)" "$gpu" "$free" "$util" "$consecutive" \
      >> "$LOG_ROOT/resource.tsv"
    if ((consecutive < 2)); then
      sleep 60
    fi
  done
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  local log="$LOG_ROOT/${scene}.log"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --scene-config "$SCENE_CONFIG" \
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
    > "$log" 2>&1
}

worker() {
  local gpu="$1"
  shift
  local scene
  for scene in "$@"; do
    wait_for_gpu "$gpu"
    if run_scene "$scene" "$gpu"; then
      printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
        >> "$LOG_ROOT/status.tsv"
    else
      local status=$?
      printf '%s\tFAILED(%s)\t%s\tGPU%s\n' \
        "$(date --iso-8601=seconds)" "$status" "$scene" "$gpu" \
        >> "$LOG_ROOT/status.tsv"
      return "$status"
    fi
  done
}

worker 2 "${GPU2_SCENES[@]}" &
PID2=$!
worker 3 "${GPU3_SCENES[@]}" &
PID3=$!

STATUS2=0
STATUS3=0
wait "$PID2" || STATUS2=$?
wait "$PID3" || STATUS3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' \
  "$(date --iso-8601=seconds)" "$STATUS2" "$STATUS3" >> "$LOG_ROOT/status.tsv"
((STATUS2 == 0 && STATUS3 == 0))
