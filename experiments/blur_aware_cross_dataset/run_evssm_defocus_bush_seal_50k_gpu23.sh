#!/usr/bin/env bash
set -Eeuo pipefail

BASE="/srv2/szha0669/blur_slam_exp"
REPO="$BASE/repos/learn2splat-official-space"
HERE="$REPO/experiments/blur_aware_cross_dataset"
PYTHON="$BASE/envs/learn2splat-py312/bin/python"
CONFIG="$HERE/scenes_deblurnerf_bush_seal_evssm.json"
RUNNER="$HERE/run_cross_dataset.py"
OUTPUT_ROOT="$BASE/outputs/learn2splat_legs_blur_evssm_defocus_bush_seal_50k_s2"
LOG_ROOT="$BASE/outputs/logs/learn2splat_legs_blur_evssm_defocus_bush_seal_50k_s2"

mkdir -p "$LOG_ROOT"
cd "$REPO"
git rev-parse HEAD > "$LOG_ROOT/code_head.txt"
git diff --binary > "$LOG_ROOT/uncommitted.patch"

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

run_scene() {
  local scene="$1"
  local gpu="$2"
  local log="$LOG_ROOT/${scene}.log"
  local output="$OUTPUT_ROOT/$scene/blur-aware"
  if [[ -s "$output/receipt.json" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
    return
  fi
  [[ ! -e "$output" ]] || { echo "refusing incomplete output: $scene" >&2; return 2; }
  wait_for_gpu "$gpu"
  printf '%s\tSTART\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --scene "$scene" --scene-config "$CONFIG" --output-root "$OUTPUT_ROOT" --device cuda:0 \
    --steps 50000 --eval-steps 10000,20000,30000,40000,50000 --seed 20260824 \
    --objective blur-aware --optimizer learned_projected --adc legs_blur --decoder-backend fastgs \
    --densification-reward off --laplacian-loss-mode surplus --laplacian-loss-weight 0.1 \
    --legs-blur-quality-weight 1.0 --legs-blur-capacity-weight 0.10 \
    --legs-blur-start-iter 2000 --legs-blur-ramp-iters 3000 \
    --legs-local-objective --coupled-dual-bpn > "$log" 2>&1
  printf '%s\tCOMPLETE\t%s\tGPU%s\n' "$(date --iso-8601=seconds)" "$scene" "$gpu" >> "$LOG_ROOT/status.tsv"
}

run_scene defocus_bush 2 & PID2=$!
run_scene defocus_seal 3 & PID3=$!
S2=0; S3=0
wait "$PID2" || S2=$?
wait "$PID3" || S3=$?
printf '%s\tQUEUE_TERMINAL\tGPU2=%s\tGPU3=%s\n' "$(date --iso-8601=seconds)" "$S2" "$S3" >> "$LOG_ROOT/status.tsv"
((S2 == 0 && S3 == 0))

"$PYTHON" - "$OUTPUT_ROOT" "$LOG_ROOT/metrics50k_summary.csv" <<'PY'
import csv, json, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
rows = []
for scene in ("defocus_bush", "defocus_seal"):
    receipt = json.loads((root / scene / "blur-aware/receipt.json").read_text())
    metric = receipt["metrics"][-1]
    rows.append({
        "scene": scene,
        "psnr_50k": metric["hold_psnr"],
        "ssim_50k": metric["hold_ssim"],
        "lpips_50k": metric["hold_lpips"],
        "evaluation_count": len(receipt["evaluation_indices"]),
    })
    if metric["hold_lpips"] is None:
        raise RuntimeError(f"LPIPS missing for {scene}")
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
PY
