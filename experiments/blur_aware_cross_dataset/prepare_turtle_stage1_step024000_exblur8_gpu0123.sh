#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv2/szha0669/blur_slam_exp
REPO="$ROOT/repos/learn2splat-official-space"
OUTPUT="$ROOT/data/turtle_stage1_step024000_exblur8_s1"
TURTLE_PYTHON=/home/szha0669/miniconda3/envs/turtle/bin/python
COMMON=(
  "$REPO/experiments/blur_aware_cross_dataset/prepare_turtle_bsd_deblurnerf.py"
  --scene-config "$REPO/experiments/blur_aware_cross_dataset/scenes_exblur8_evssm.generated.json"
  --scene-prefix exblur_
  --output-root "$OUTPUT"
  --turtle-repo "$ROOT/repos/Turtle"
  --checkpoint /home/szha0669/C3G/step_024000.pth
  --checkpoint-state-key model
  --config "$ROOT/repos/Turtle/options/Turtle_Deblur_Gopro.yml"
  --model-type t1
  --linear-rgb
  --prediction-only
  --runner "$REPO/experiments/blur_aware_cross_dataset/run_turtle_no_gt_cuda.py"
  --python "$TURTLE_PYTHON"
)

mkdir -p "$OUTPUT/logs"

run_pair() {
  local gpu=$1
  local first=$2
  local second=$3
  CUDA_VISIBLE_DEVICES="$gpu" "$TURTLE_PYTHON" "${COMMON[@]}" \
    --scene "$first" --scene "$second" \
    >"$OUTPUT/logs/${first}_${second}.log" 2>&1
}

run_pair 0 exblur_bench exblur_camellia & pid0=$!
run_pair 1 exblur_dragon exblur_jars & pid1=$!
run_pair 2 exblur_jars2 exblur_postbox & pid2=$!
run_pair 3 exblur_stone_lantern exblur_sunflowers & pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  wait "$pid" || status=$?
done
exit "$status"
