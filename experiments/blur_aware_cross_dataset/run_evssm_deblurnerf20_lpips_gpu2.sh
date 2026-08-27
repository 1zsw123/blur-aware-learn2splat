#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
FULL3="$BASE/outputs/learn2splat_legs_blur_full3_50k_local_joint_s1"
REMAINING="$BASE/outputs/learn2splat_legs_blur_deblurnerf_remaining18_50k_local_joint_s2"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_evssm_deblurnerf20_lpips50k_s1"
POST="$HERE/postprocess_lpips_from_ply.py"
mkdir -p "$LOG_ROOT"
cd "$REPO"

run_one() {
  local scene="$1" root="$2" config="$3"
  local output="$root/$scene/blur-aware/lpips_postprocess.json"
  if [[ -s "$output" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
    return
  fi
  printf '%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
  if CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$POST" --scene "$scene" --run-root "$root" \
    --scene-config "$config" --device cuda:0 > "$LOG_ROOT/${scene}.log" 2>&1; then
    printf '%s\tCOMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" >> "$LOG_ROOT/status.tsv"
  else
    local status=$?
    printf '%s\tFAILED(%s)\t%s\n' "$(date --iso-8601=seconds)" "$status" "$scene" >> "$LOG_ROOT/status.tsv"
  fi
}

run_one motion_blurcoffee "$FULL3" "$HERE/scenes.json"
run_one defocus_cisco "$FULL3" "$HERE/scenes.json"

mapfile -t SCENES < <("$PYTHON" -c "import json; d=json.load(open('$HERE/scenes_deblurnerf_remaining18.json')); print(*sorted(d), sep='\\n')")
for scene in "${SCENES[@]}"; do
  run_one "$scene" "$REMAINING" "$HERE/scenes_deblurnerf_remaining18.json"
done

"$PYTHON" - "$FULL3" "$REMAINING" "$LOG_ROOT/lpips50k_summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path
full3, remaining, output = map(Path, sys.argv[1:])
rows = []
for root in (full3, remaining):
    for path in sorted(root.glob("*/blur-aware/lpips_postprocess.json")):
        payload = json.loads(path.read_text())
        scene = payload["scene"]
        if not scene.startswith(("motion_", "defocus_")):
            continue
        rows.append({
            "scene": scene,
            "dataset": "Motion10" if scene.startswith("motion_") else "Defocus10",
            "lpips_50k": payload["hold_lpips"],
            "psnr_50k": payload["rerender_psnr"],
            "ssim_50k": payload["rerender_ssim"],
            "evaluation_count": payload["evaluation_count"],
        })
if not rows:
    raise RuntimeError("no LPIPS results were produced")
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: row["scene"]))
for dataset in ("Motion10", "Defocus10"):
    values = [float(row["lpips_50k"]) for row in rows if row["dataset"] == dataset]
    print(dataset, sum(values) / len(values))
print("Combined20", sum(float(row["lpips_50k"]) for row in rows) / len(rows))
PY
