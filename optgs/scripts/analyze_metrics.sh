#!/usr/bin/env bash
# Summarize per-scene metrics under <optimizer_metrics_dir>/<scene>/target_<optimizer>.json
# and (optionally) the PSNR improvement over an init-only baseline.
#
# Usage:
#   analyze_metrics.sh <optimizer_metrics_dir> [init_only_metrics_dir]
#
# Examples:
#   analyze_metrics.sh \
#     results/mipnerf360/8_all_9_scenes/3dgslm_ply_init/vanilla/100_match_3dgslm_metrics/vanillaoptimizer/metrics
#
#   analyze_metrics.sh \
#     results/mipnerf360_sfm/.../clogsoptimizer/metrics \
#     results/mipnerf360_sfm/8_all_9_scenes/3dgslm_ply_init/init_only/100_match_3dgslm_metrics/initializerply/metrics

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <optimizer_metrics_dir> [init_only_metrics_dir]" >&2
    exit 1
fi

OPT_DIR="$1"
INIT_DIR="${2:-}"

python3 - "$OPT_DIR" "$INIT_DIR" <<'PY'
import sys, os, glob, json

opt_dir = sys.argv[1]
init_dir = sys.argv[2] or None

def find_target_json(metrics_dir):
    # Expect <metrics_dir>/<scene>/target_<module>.json
    files = sorted(glob.glob(os.path.join(metrics_dir, "*", "target_*.json")))
    if not files:
        sys.exit(f"No target_*.json found under {metrics_dir}")
    module = os.path.basename(files[0])[len("target_"):-len(".json")]
    for f in files:
        if not os.path.basename(f) == f"target_{module}.json":
            sys.exit(f"Mixed module names under {metrics_dir}: {f}")
    return files, module

def find_any(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            yield obj[key]
        for v in obj.values():
            yield from find_any(v, key)
    elif isinstance(obj, list):
        for x in obj:
            yield from find_any(x, key)

opt_files, opt_mod = find_target_json(opt_dir)
opt_psnr, opt_vram, opt_time_s, opt_iters, opt_gauss = {}, {}, {}, {}, {}
for f in opt_files:
    scene = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        d = json.load(fh)
    psnr_list = d.get(f"{opt_mod}_psnr")
    time_list = d.get(f"{opt_mod}_time")
    iters_list = d.get(f"{opt_mod}_iterations")
    gauss_list = d.get(f"{opt_mod}_gaussians")
    vram = list(find_any(d, "peak_vram_mb"))
    opt_psnr[scene]   = psnr_list[-1]  if psnr_list else None
    opt_time_s[scene] = time_list[-1]/1000.0 if time_list else None
    opt_iters[scene]  = iters_list[-1] if iters_list else None
    opt_gauss[scene]  = gauss_list[-1] if gauss_list else None
    opt_vram[scene]   = vram[0] if vram else None

init_psnr = {}
init_mod = None
if init_dir:
    init_files, init_mod = find_target_json(init_dir)
    for f in init_files:
        scene = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            d = json.load(fh)
        psnr_list = d.get(f"{init_mod}_psnr")
        init_psnr[scene] = psnr_list[-1] if psnr_list else None

print(f"Optimizer module: {opt_mod}   (dir: {opt_dir})")
if init_mod:
    print(f"Init module     : {init_mod}   (dir: {init_dir})")
print()

scenes = sorted(opt_psnr)
header = f"{'scene':<10}"
if init_dir:
    header += f" {'init':>8} {'last':>8} {'diff':>8}"
else:
    header += f" {'last_psnr':>10}"
header += f" {'vram_MB':>10} {'time_s':>9} {'iters':>7} {'gauss':>10}"
print(header)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

diffs = []
for s in scenes:
    row = f"{s:<10}"
    if init_dir:
        ip = init_psnr.get(s)
        op = opt_psnr.get(s)
        diff = (op - ip) if (ip is not None and op is not None) else None
        if diff is not None:
            diffs.append(diff)
        row += f" {('--' if ip is None else f'{ip:8.3f}')}"
        row += f" {('--' if op is None else f'{op:8.3f}')}"
        row += f" {('--' if diff is None else f'{diff:+8.3f}')}"
    else:
        op = opt_psnr.get(s)
        row += f" {('--' if op is None else f'{op:10.3f}')}"
    v = opt_vram.get(s);   row += f" {('--' if v is None else f'{v:10.2f}')}"
    t = opt_time_s.get(s); row += f" {('--' if t is None else f'{t:9.3f}')}"
    it = opt_iters.get(s); row += f" {('--' if it is None else f'{it:7d}')}"
    g = opt_gauss.get(s);  row += f" {('--' if g is None else f'{g:10d}')}"
    print(row)

print()
def fmt(x, spec, suffix=""):
    return "--" if x is None else f"{x:{spec}}{suffix}"

print(f"n scenes        : {len(scenes)}")
print(f"avg last_psnr   : {fmt(mean(opt_psnr.values()), '.3f')}")
if init_dir:
    print(f"avg init_psnr   : {fmt(mean(init_psnr.values()), '.3f')}")
    print(f"avg improvement : {fmt(mean(diffs) if diffs else None, '+.3f', ' dB')}")
print(f"avg peak_vram   : {fmt(mean(opt_vram.values()), '.2f', ' MB')}")
print(f"avg time        : {fmt(mean(opt_time_s.values()), '.3f', ' s')}")
print(f"avg iters       : {fmt(mean(opt_iters.values()), '.1f')}")
print(f"avg gaussians   : {fmt(mean(opt_gauss.values()), '.0f')}")
PY
