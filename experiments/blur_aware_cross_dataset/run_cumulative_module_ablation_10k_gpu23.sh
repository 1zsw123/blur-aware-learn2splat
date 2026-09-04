#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/srv2/szha0669/blur_slam_exp/repos/learn2splat-official-space"
PYTHON="/srv2/szha0669/blur_slam_exp/envs/learn2splat-py312/bin/python"
RUNNER="$REPO/experiments/blur_aware_cross_dataset/run_cross_dataset.py"
RUN_TAG="${RUN_TAG:-cumulative_modules_10k_s1}"
OUTPUT_ROOT="/srv2/szha0669/blur_slam_exp/outputs/learn2splat_blur_legs_${RUN_TAG}"
LOG_ROOT="/srv2/szha0669/blur_slam_exp/outputs/logs/learn2splat_blur_legs_${RUN_TAG}"
SEED="${SEED:-20260830}"

SCENES=(motion_blurcoffee defocus_cisco tum_fr1_desk)
STAGES=(m0_base m1_coupled_bpn m2_laplacian m3_blur_state m4_full_blur_legs)
GPUS=(2 3)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO"

git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

cat > "$LOG_ROOT/ablation_contract.tsv" <<'EOF'
stage	objective	coupled_dual_bpn	laplacian_surplus	blur_conditioned_state	blur_local_objective	global_quality_reward	capacity_cost
m0_base	photometric	NO	NO	NO	NO	NO	NO
m1_coupled_bpn	blur-aware	YES	NO	NO	NO	NO	NO
m2_laplacian	blur-aware	YES	YES	NO	NO	NO	NO
m3_blur_state	blur-aware	YES	YES	YES	YES	NO	NO
m4_full_blur_legs	blur-aware	YES	YES	YES	YES	YES	YES
EOF

for stage in "${STAGES[@]}"; do
  for scene in "${SCENES[@]}"; do
    if [[ -e "$OUTPUT_ROOT/$stage/$scene" ]]; then
      echo "refusing to overwrite existing output: $OUTPUT_ROOT/$stage/$scene" >&2
      exit 2
    fi
  done
done

stage_args() {
  case "$1" in
    m0_base)
      printf '%s\0' --objective photometric --adc legs --decoder-backend fastgs \
        --densification-reward off --laplacian-loss-weight 0
      ;;
    m1_coupled_bpn)
      printf '%s\0' --objective blur-aware --adc legs --decoder-backend fastgs \
        --densification-reward off --laplacian-loss-weight 0 --coupled-dual-bpn
      ;;
    m2_laplacian)
      printf '%s\0' --objective blur-aware --adc legs --decoder-backend fastgs \
        --densification-reward off --laplacian-loss-mode surplus \
        --laplacian-loss-weight 0.1 --coupled-dual-bpn
      ;;
    m3_blur_state)
      printf '%s\0' --objective blur-aware --adc legs_blur --decoder-backend fastgs \
        --densification-reward off --laplacian-loss-mode surplus \
        --laplacian-loss-weight 0.1 --coupled-dual-bpn --legs-local-objective \
        --legs-blur-quality-weight 0 --legs-blur-capacity-weight 0 \
        --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000
      ;;
    m4_full_blur_legs)
      printf '%s\0' --objective blur-aware --adc legs_blur --decoder-backend fastgs \
        --densification-reward off --laplacian-loss-mode surplus \
        --laplacian-loss-weight 0.1 --coupled-dual-bpn --legs-local-objective \
        --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
        --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000
      ;;
    *) return 2 ;;
  esac
}

JOBS=()
for stage in "${STAGES[@]}"; do
  for scene in "${SCENES[@]}"; do
    JOBS+=("$stage|$scene")
  done
done

declare -A PID_GPU=()
declare -A PID_JOB=()
PIDS=()
NEXT_JOB=0
FAILED=0

launch_job() {
  local job="$1" gpu="$2" stage scene log stage_root
  IFS='|' read -r stage scene <<< "$job"
  stage_root="$OUTPUT_ROOT/$stage"
  log="$LOG_ROOT/${stage}__${scene}.log"
  mkdir -p "$stage_root"
  mapfile -d '' -t EXTRA < <(stage_args "$stage")
  printf '%s\tSTART\t%s\t%s\tGPU%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" \
    --output-root "$stage_root" \
    --device cuda:0 \
    --steps 10000 \
    --eval-steps 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
    --seed "$SEED" \
    --optimizer learned_projected \
    "${EXTRA[@]}" \
    > "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_JOB["$pid"]="$job"
  NEXT_JOB=$((NEXT_JOB + 1))
}

for gpu in "${GPUS[@]}"; do
  launch_job "${JOBS[$NEXT_JOB]}" "$gpu"
done

while ((${#PIDS[@]} > 0)); do
  DONE_PID=""
  set +e
  wait -n -p DONE_PID "${PIDS[@]}"
  STATUS=$?
  set -e
  [[ -n "$DONE_PID" ]] || exit 3
  GPU="${PID_GPU[$DONE_PID]}"
  JOB="${PID_JOB[$DONE_PID]}"
  IFS='|' read -r STAGE SCENE <<< "$JOB"
  REMAINING=()
  for pid in "${PIDS[@]}"; do
    [[ "$pid" == "$DONE_PID" ]] || REMAINING+=("$pid")
  done
  PIDS=("${REMAINING[@]}")
  if ((STATUS == 0)); then
    printf '%s\tCOMPLETE\t%s\t%s\tGPU%s\n' \
      "$(date --iso-8601=seconds)" "$STAGE" "$SCENE" "$GPU" >> "$LOG_ROOT/status.tsv"
    if ((FAILED == 0 && NEXT_JOB < ${#JOBS[@]})); then
      launch_job "${JOBS[$NEXT_JOB]}" "$GPU"
    fi
  else
    printf '%s\tFAILED(%s)\t%s\t%s\tGPU%s\n' \
      "$(date --iso-8601=seconds)" "$STATUS" "$STAGE" "$SCENE" "$GPU" >> "$LOG_ROOT/status.tsv"
    FAILED=1
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
done

((FAILED == 0 && NEXT_JOB == ${#JOBS[@]}))
