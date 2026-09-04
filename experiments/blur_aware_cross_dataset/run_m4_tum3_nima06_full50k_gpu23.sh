#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
RUN_TAG="${RUN_TAG:-m4_nima06_full50k_s1}"
METHOD_LABEL="${METHOD_LABEL:-M4 full Blur-LeGS}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-1.0}"
CAPACITY_WEIGHT="${CAPACITY_WEIGHT:-0.10}"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_tum3_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_legs_blur_tum3_${RUN_TAG}"
SEED="${SEED:-20260830}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

for scene in tum_fr1_desk tum_fr2_xyz tum_fr3_office; do
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing to overwrite $OUTPUT_ROOT/$scene/blur-aware" >&2
    exit 2
  }
done

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
printf '%s; quality_weight=%s; capacity_weight=%s; TUM/I2-SLAM mapping protocol;\n' \
  "$METHOD_LABEL" "$QUALITY_WEIGHT" "$CAPACITY_WEIGHT" > "$LOG_ROOT/contract.txt"
printf '%s\n' \
  'existing NIMA>0.6 sharp/w10 contract unchanged; 50K; evaluations at 10K/20K/30K/40K/50K; LPIPS enabled.' \
  >> "$LOG_ROOT/contract.txt"

wait_for_gpu3_cumulative() {
  while pgrep -af run_cross_dataset.py \
    | grep -F 'cumulative_modules_10k_s1/m4_full_blur_legs' \
    | grep -F 'tum_fr1_desk' >/dev/null; do
    sleep 30
  done
}

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_ROOT/${scene}.log"
  local telemetry="$LOG_ROOT/${scene}_gpu_telemetry.tsv"
  local start_epoch end_epoch pid peak=0 used
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
  start_epoch=$(date +%s)
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --steps 50000 \
    --eval-steps 10000,20000,30000,40000,50000 \
    --seed "$SEED" \
    --objective blur-aware \
    --optimizer learned_projected \
    --adc legs_blur \
    --decoder-backend fastgs \
    --densification-reward off \
    --laplacian-loss-mode surplus \
    --laplacian-loss-weight 0.1 \
    --coupled-dual-bpn \
    --legs-local-objective \
    --legs-blur-quality-weight "$QUALITY_WEIGHT" \
    --legs-blur-capacity-weight "$CAPACITY_WEIGHT" \
    --legs-blur-start-iter 2000 \
    --legs-blur-ramp-iters 3000 \
    > "$log" 2>&1 &
  pid=$!
  printf 'utc\tmemory_used_mib\tutilization_gpu_pct\n' > "$telemetry"
  while kill -0 "$pid" 2>/dev/null; do
    read -r used util < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        -i "$gpu" | tr -d ','
    )
    ((used > peak)) && peak=$used
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$used" "$util" >> "$telemetry"
    sleep 10
  done
  set +e
  wait "$pid"
  local status=$?
  set -e
  end_epoch=$(date +%s)
  printf 'elapsed_seconds=%s\npeak_memory_used_mib=%s\n' \
    "$((end_epoch - start_epoch))" "$peak" > "$LOG_ROOT/${scene}_runtime.txt"
  if ((status != 0)); then
    printf '%s\tFAILED(%s)\t%s\tGPU%s\n' \
      "$(date --iso-8601=seconds)" "$status" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
    return "$status"
  fi
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" \
    >> "$LOG_ROOT/status.tsv"
}

worker_gpu2() {
  run_scene tum_fr2_xyz 2
  run_scene tum_fr3_office 2
}

worker_gpu3() {
  wait_for_gpu3_cumulative
  run_scene tum_fr1_desk 3
}

worker_gpu2 &
PID2=$!
worker_gpu3 &
PID3=$!
set +e
wait "$PID2"; STATUS2=$?
wait "$PID3"; STATUS3=$?
set -e
((STATUS2 == 0 && STATUS3 == 0))
