#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
TURTLE_PYTHON="/home/szha0669/miniconda3/envs/turtle/bin/python"
TURTLE_REPO="$BASE/repos/Turtle"
CORE_CONFIG="$HERE/scenes.json"
REMAINING_CONFIG="$HERE/scenes_deblurnerf_remaining18.json"
TEACHER_ROOT="$BASE/data/turtle_bsd_deblurred_deblurnerf"
SCENE_CONFIG="$HERE/scenes_deblurnerf20_turtle_bsd.generated.json"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_turtle_bsd_deblurnerf20_50k_s1"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_turtle_bsd_deblurnerf20_50k_s1"
RUNNER="$HERE/run_cross_dataset.py"
CKPT="$BASE/checkpoints/turtle/Turtle_models/BSD_Deblur.pth"
TURTLE_CONFIG="$TURTLE_REPO/options/Turtle_Derain_VRDS.yml"

mkdir -p "$LOG_ROOT"
cd "$REPO"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
sha256sum "$CKPT" > "$LOG_ROOT/turtle_bsd_checkpoint.sha256"

wait_for_gpu() {
  local gpu="$1" consecutive=0
  while ((consecutive < 2)); do
    local free util
    IFS=',' read -r free util < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if ((free >= 30000 && (gpu == 2 || util <= 10))); then consecutive=$((consecutive + 1)); else consecutive=0; fi
    printf '%s\tWAIT_GPU\tGPU%s\tfree_mb=%s\tutil=%s\tstable=%s/2\n' "$(date --iso-8601=seconds)" "$gpu" "$free" "$util" "$consecutive" >> "$LOG_ROOT/resource.tsv"
    ((consecutive == 2)) || sleep 60
  done
}

# Turtle is causal and all twenty sequences are generated in one model load.
TEACHER_COMPLETE_COUNT="$(find "$TEACHER_ROOT" -mindepth 2 -maxdepth 2 -name .complete.json 2>/dev/null | wc -l)"
if [[ "$TEACHER_COMPLETE_COUNT" -ne 20 ]]; then
  wait_for_gpu 3
  CUDA_VISIBLE_DEVICES=3 "$TURTLE_PYTHON" "$HERE/prepare_turtle_bsd_deblurnerf.py" \
    --scene-config "$CORE_CONFIG" --scene-config "$REMAINING_CONFIG" \
    --output-root "$TEACHER_ROOT" --turtle-repo "$TURTLE_REPO" \
    --checkpoint "$CKPT" --config "$TURTLE_CONFIG" \
    --runner "$HERE/run_turtle_no_gt_cuda.py" --python "$TURTLE_PYTHON" \
    > "$LOG_ROOT/turtle_teacher_generation.log" 2>&1
else
  printf '%s\tTEACHERS_COMPLETE\tcount=20\n' "$(date --iso-8601=seconds)" >> "$LOG_ROOT/status.tsv"
fi

"$PYTHON" - "$CORE_CONFIG" "$REMAINING_CONFIG" "$TEACHER_ROOT" "$SCENE_CONFIG" <<'PY'
import json, sys
core, remaining, teacher_root, destination = sys.argv[1:]
scenes = {}
for path in (core, remaining):
    with open(path, encoding="utf-8") as handle:
        scenes.update(json.load(handle))
scenes = {k: v for k, v in scenes.items() if k.startswith(("motion_", "defocus_"))}
if len(scenes) != 20:
    raise RuntimeError(f"expected 20 scenes, found {len(scenes)}")
for name, cfg in scenes.items():
    cfg["evssm_dir"] = f"{teacher_root}/{name}"
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(scenes, handle, indent=2)
    handle.write("\n")
PY

mapfile -t MOTION < <("$PYTHON" -c "import json; d=json.load(open('$SCENE_CONFIG')); print(*sorted(k for k in d if k.startswith('motion_')), sep='\\n')")
mapfile -t DEFOCUS < <("$PYTHON" -c "import json; d=json.load(open('$SCENE_CONFIG')); print(*sorted(k for k in d if k.startswith('defocus_')), sep='\\n')")
GPU2_SCENES=("${MOTION[@]:0:5}" "${DEFOCUS[@]:0:5}")
GPU3_SCENES=("${MOTION[@]:5:5}" "${DEFOCUS[@]:5:5}")

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_ROOT/${scene}.log"
  if [[ -s "$OUTPUT_ROOT/$scene/blur-aware/receipt.json" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
    return
  fi
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || { echo "refusing incomplete output: $scene" >&2; return 2; }
  wait_for_gpu "$gpu"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$SCENE_CONFIG" --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 50000 --eval-steps 10000,20000,30000,40000,50000 --seed 20260824 \
    --objective blur-aware --optimizer learned_projected --adc legs_blur --decoder-backend fastgs \
    --densification-reward off --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    --legs-local-objective --coupled-dual-bpn --skip-lpips > "$log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

worker() { local gpu="$1"; shift; local scene; for scene in "$@"; do run_scene "$scene" "$gpu" || return $?; done; }
worker 2 "${GPU2_SCENES[@]}" & PID2=$!
worker 3 "${GPU3_SCENES[@]}" & PID3=$!
S2=0; S3=0
wait "$PID2" || S2=$?
wait "$PID3" || S3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$S2" "$S3" >> "$LOG_ROOT/status.tsv"
((S2 == 0 && S3 == 0))
