#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
TURTLE_PYTHON="/home/szha0669/miniconda3/envs/turtle/bin/python"
TURTLE_REPO="$BASE/repos/Turtle"
CORE_CONFIG="$HERE/scenes.json"
TEACHER_ROOT="$BASE/data/turtle_bsd_deblurred_tum3"
SCENE_CONFIG="$HERE/scenes_tum3_turtle_bsd.generated.json"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_turtle_bsd_tum3_50k_s1"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_turtle_bsd_tum3_50k_s1"
LPIPS_LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_turtle_bsd_tum3_lpips50k_s1"
RUNNER="$HERE/run_cross_dataset.py"
PREPARER="$HERE/prepare_turtle_bsd_deblurnerf.py"
POST="$HERE/postprocess_lpips_from_ply.py"
CKPT="$BASE/checkpoints/turtle/Turtle_models/BSD_Deblur.pth"
TURTLE_CONFIG="$TURTLE_REPO/options/Turtle_Derain_VRDS.yml"
SCENES=(tum_fr1_desk tum_fr2_xyz tum_fr3_office)

mkdir -p "$LOG_ROOT" "$LPIPS_LOG_ROOT"
cd "$REPO"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"
sha256sum "$CKPT" > "$LOG_ROOT/turtle_bsd_checkpoint.sha256"

wait_for_gpu() {
  local gpu="$1" consecutive=0
  while ((consecutive < 2)); do
    local free util
    IFS=',' read -r free util < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if ((free >= 30000 && util <= 10)); then consecutive=$((consecutive + 1)); else consecutive=0; fi
    printf '%s\tWAIT_GPU\tGPU%s\tfree_mb=%s\tutil=%s\tstable=%s/2\n' "$(date --iso-8601=seconds)" "$gpu" "$free" "$util" "$consecutive" >> "$LOG_ROOT/resource.tsv"
    ((consecutive == 2)) || sleep 60
  done
}

prepare_group() {
  local gpu="$1"; shift
  wait_for_gpu "$gpu"
  local args=()
  local scene
  for scene in "$@"; do args+=(--scene "$scene"); done
  CUDA_VISIBLE_DEVICES="$gpu" "$TURTLE_PYTHON" "$PREPARER" \
    --scene-config "$CORE_CONFIG" --scene-prefix tum_ --output-naming index \
    "${args[@]}" --output-root "$TEACHER_ROOT" --turtle-repo "$TURTLE_REPO" \
    --checkpoint "$CKPT" --config "$TURTLE_CONFIG" \
    --runner "$HERE/run_turtle_no_gt_cuda.py" --python "$TURTLE_PYTHON" \
    > "$LOG_ROOT/turtle_teacher_gpu${gpu}.log" 2>&1
}

prepare_group 2 tum_fr1_desk tum_fr2_xyz & TPID2=$!
prepare_group 3 tum_fr3_office & TPID3=$!
TS2=0; TS3=0
wait "$TPID2" || TS2=$?
wait "$TPID3" || TS3=$?
printf '%s\tTEACHER_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$TS2" "$TS3" >> "$LOG_ROOT/status.tsv"
((TS2 == 0 && TS3 == 0))

"$PYTHON" - "$CORE_CONFIG" "$TEACHER_ROOT" "$SCENE_CONFIG" <<'PY'
import json, sys
source, teacher_root, destination = sys.argv[1:]
all_scenes = json.load(open(source, encoding="utf-8"))
names = ("tum_fr1_desk", "tum_fr2_xyz", "tum_fr3_office")
scenes = {name: all_scenes[name] for name in names}
for name, cfg in scenes.items():
    cfg["evssm_dir"] = f"{teacher_root}/{name}"
    cfg["teacher_model"] = "Turtle BSD_Deblur"
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(scenes, handle, indent=2)
    handle.write("\n")
PY

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
worker 2 tum_fr1_desk tum_fr2_xyz & PID2=$!
worker 3 tum_fr3_office & PID3=$!
S2=0; S3=0
wait "$PID2" || S2=$?
wait "$PID3" || S3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$S2" "$S3" >> "$LOG_ROOT/status.tsv"
((S2 == 0 && S3 == 0))

for scene in "${SCENES[@]}"; do
  output="$OUTPUT_ROOT/$scene/blur-aware/lpips_postprocess.json"
  if [[ -s "$output" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LPIPS_LOG_ROOT/status.tsv"
    continue
  fi
  wait_for_gpu 2
  printf '%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LPIPS_LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$POST" --scene "$scene" --run-root "$OUTPUT_ROOT" \
    --scene-config "$SCENE_CONFIG" --device cuda:0 > "$LPIPS_LOG_ROOT/${scene}.log" 2>&1
  printf '%s\tCOMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LPIPS_LOG_ROOT/status.tsv"
done

"$PYTHON" - "$OUTPUT_ROOT" "$LPIPS_LOG_ROOT/lpips50k_summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(root.glob("*/blur-aware/lpips_postprocess.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "scene": payload["scene"],
        "lpips_50k": payload["hold_lpips"],
        "psnr_50k": payload["rerender_psnr"],
        "ssim_50k": payload["rerender_ssim"],
        "evaluation_count": payload["evaluation_count"],
    })
if len(rows) != 3:
    raise RuntimeError(f"expected three TUM LPIPS rows, got {len(rows)}")
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("TUM3", sum(float(r["psnr_50k"]) for r in rows) / 3,
      sum(float(r["ssim_50k"]) for r in rows) / 3,
      sum(float(r["lpips_50k"]) for r in rows) / 3)
PY
