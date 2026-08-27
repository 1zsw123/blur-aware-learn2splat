#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
RUN_ROOT="$BASE/outputs/learn2splat_legs_blur_turtle_bsd_deblurnerf20_50k_s1"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_turtle_bsd_deblurnerf20_lpips50k_s1"
SCENES="$HERE/scenes_deblurnerf20_turtle_bsd.generated.json"
POST="$HERE/postprocess_lpips_from_ply.py"
mkdir -p "$LOG_ROOT"
cd "$REPO"

mapfile -t NAMES < <("$PYTHON" -c "import json; d=json.load(open('$SCENES')); print(*sorted(d), sep='\\n')")

for scene in "${NAMES[@]}"; do
  receipt="$RUN_ROOT/$scene/blur-aware/receipt.json"
  output="$RUN_ROOT/$scene/blur-aware/lpips_postprocess.json"
  while [[ ! -s "$receipt" ]]; do
    printf '%s\tWAIT_TRAINING\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
    sleep 300
  done
  if [[ -s "$output" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
    continue
  fi
  printf '%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
  if CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$POST" --scene "$scene" --run-root "$RUN_ROOT" \
    --scene-config "$SCENES" --device cuda:0 > "$LOG_ROOT/${scene}.log" 2>&1; then
    printf '%s\tCOMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
  else
    status=$?
    printf '%s\tFAILED(%s)\t%s\n' "$(date --iso-8601=seconds)" "$status" "$scene" >> "$LOG_ROOT/status.tsv"
  fi
done

"$PYTHON" - "$RUN_ROOT" "$LOG_ROOT/lpips50k_summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(root.glob("*/blur-aware/lpips_postprocess.json")):
    payload = json.loads(path.read_text())
    scene = payload["scene"]
    rows.append({
        "scene": scene,
        "dataset": "Motion10" if scene.startswith("motion_") else "Defocus10",
        "lpips_50k": payload["hold_lpips"],
        "psnr_50k": payload["rerender_psnr"],
        "ssim_50k": payload["rerender_ssim"],
        "evaluation_count": payload["evaluation_count"],
    })
if not rows:
    raise RuntimeError("no Turtle BSD LPIPS results were produced")
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
for dataset in ("Motion10", "Defocus10"):
    values = [float(row["lpips_50k"]) for row in rows if row["dataset"] == dataset]
    print(dataset, len(values), sum(values) / len(values))
values = [float(row["lpips_50k"]) for row in rows]
print("Combined20", len(values), sum(values) / len(values))
PY
