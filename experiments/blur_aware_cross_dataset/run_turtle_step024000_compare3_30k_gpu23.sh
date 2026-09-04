#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
TURTLE_PYTHON="/home/szha0669/miniconda3/envs/turtle/bin/python"
TURTLE_REPO="$BASE/repos/Turtle"
CKPT="/home/szha0669/C3G/step_024000.pth"
TURTLE_CONFIG="$TURTLE_REPO/options/Turtle_Deblur_Gopro.yml"
CORE_CONFIG="$HERE/scenes.json"
SCENE_CONFIG="$HERE/scenes_turtle_step024000_compare3.generated.json"
TEACHER_ROOT="$BASE/data/turtle_stage1_step024000_compare3_s1"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_turtle_step024000_compare3_30k_s1"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_turtle_step024000_compare3_30k_s1"
EVSSM_ROOT="$BASE/outputs/learn2splat_legs_blur_full3_50k_local_joint_s1"
PREPARER="$HERE/prepare_turtle_bsd_deblurnerf.py"
TURTLE_RUNNER="$HERE/run_turtle_no_gt_cuda.py"
RUNNER="$HERE/run_cross_dataset.py"

mkdir -p "$LOG_ROOT"
cd "$REPO"
sha256sum "$CKPT" > "$LOG_ROOT/turtle_checkpoint.sha256"
sha256sum "$TURTLE_CONFIG" > "$LOG_ROOT/turtle_config.sha256"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

# Do not race the already authorized ExBlur workers between scene boundaries.
while tmux has-session -t exblur8_evssm_legs50k_s1 2>/dev/null; do
  printf '%s\tWAIT_EXBLUR_QUEUE\n' "$(date --iso-8601=seconds)" >> "$LOG_ROOT/status.tsv"
  sleep 120
done

prepare_deblurnerf() {
  CUDA_VISIBLE_DEVICES=2 "$TURTLE_PYTHON" "$PREPARER" \
    --scene-config "$CORE_CONFIG" \
    --scene motion_blurcoffee --scene defocus_cisco \
    --output-root "$TEACHER_ROOT" --turtle-repo "$TURTLE_REPO" \
    --checkpoint "$CKPT" --checkpoint-state-key model \
    --config "$TURTLE_CONFIG" --model-type t1 --linear-rgb --prediction-only \
    --runner "$TURTLE_RUNNER" --python "$TURTLE_PYTHON" \
    > "$LOG_ROOT/turtle_teacher_motion_defocus.log" 2>&1
}

prepare_tum() {
  CUDA_VISIBLE_DEVICES=3 "$TURTLE_PYTHON" "$PREPARER" \
    --scene-config "$CORE_CONFIG" --scene-prefix tum_ --scene tum_fr2_xyz \
    --output-naming index \
    --output-root "$TEACHER_ROOT" --turtle-repo "$TURTLE_REPO" \
    --checkpoint "$CKPT" --checkpoint-state-key model \
    --config "$TURTLE_CONFIG" --model-type t1 --linear-rgb --prediction-only \
    --runner "$TURTLE_RUNNER" --python "$TURTLE_PYTHON" \
    > "$LOG_ROOT/turtle_teacher_tum.log" 2>&1
}

prepare_deblurnerf & TPID2=$!
prepare_tum & TPID3=$!
TS2=0; TS3=0
wait "$TPID2" || TS2=$?
wait "$TPID3" || TS3=$?
printf '%s\tTEACHER_TERMINAL\tGPU2=%s\tGPU3=%s\n' \
  "$(date --iso-8601=seconds)" "$TS2" "$TS3" >> "$LOG_ROOT/status.tsv"
((TS2 == 0 && TS3 == 0))

"$PYTHON" - "$CORE_CONFIG" "$TEACHER_ROOT" "$SCENE_CONFIG" <<'PY'
import json, sys
source, teacher_root, destination = sys.argv[1:]
all_scenes = json.load(open(source, encoding="utf-8"))
names = ("motion_blurcoffee", "defocus_cisco", "tum_fr2_xyz")
scenes = {name: all_scenes[name] for name in names}
for name, cfg in scenes.items():
    cfg["evssm_dir"] = f"{teacher_root}/{name}"
    cfg["teacher_model"] = "Turtle Stage1 step_024000"
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(scenes, handle, indent=2); handle.write("\n")
PY

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_ROOT/${scene}.log"
  [[ ! -e "$OUTPUT_ROOT/$scene/blur-aware" ]] || {
    echo "refusing existing output: $OUTPUT_ROOT/$scene/blur-aware" >&2
    return 2
  }
  printf '%s\tTRAIN_START\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$SCENE_CONFIG" --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 30000 --eval-steps 10000,20000,30000 --seed 20260824 \
    --objective blur-aware --optimizer learned_projected --adc legs_blur --decoder-backend fastgs \
    --densification-reward off --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    --legs-local-objective --coupled-dual-bpn \
    > "$log" 2>&1
  printf '%s\tTRAIN_COMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

(run_scene motion_blurcoffee 2 && run_scene defocus_cisco 2) & PID2=$!
run_scene tum_fr2_xyz 3 & PID3=$!
S2=0; S3=0
wait "$PID2" || S2=$?
wait "$PID3" || S3=$?
printf '%s\tTRAIN_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$S2" "$S3" >> "$LOG_ROOT/status.tsv"
((S2 == 0 && S3 == 0))

"$PYTHON" - "$OUTPUT_ROOT" "$EVSSM_ROOT" "$LOG_ROOT/compare_evssm_at30k.csv" <<'PY'
import csv, json, sys
from pathlib import Path
turtle, evssm, output = map(Path, sys.argv[1:])
rows = []
for scene in ("motion_blurcoffee", "defocus_cisco", "tum_fr2_xyz"):
    t = json.loads((turtle / scene / "blur-aware/receipt.json").read_text())
    e = json.loads((evssm / scene / "blur-aware/receipt.json").read_text())
    tm = next(row for row in t["metrics"] if row["step"] == 30000)
    em = next(row for row in e["metrics"] if row["step"] == 30000)
    rows.append({
        "scene": scene,
        "turtle_step24k_psnr_30k": tm["hold_psnr"],
        "evssm_psnr_30k": em["hold_psnr"],
        "delta_psnr": tm["hold_psnr"] - em["hold_psnr"],
        "turtle_step24k_ssim_30k": tm["hold_ssim"],
        "evssm_ssim_30k": em["hold_ssim"],
        "delta_ssim": tm["hold_ssim"] - em["hold_ssim"],
        "turtle_step24k_lpips_30k": tm["hold_lpips"],
        "evssm_lpips_30k": em.get("hold_lpips"),
    })
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
