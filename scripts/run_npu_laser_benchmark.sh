#!/usr/bin/env bash
# Usage: ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_laser_benchmark.sh
#   /path/to/laser_benchmark/v1 checkpoints/MEOW-sB25.pth work/laser-benchmark/run [blk_mixed ...]
set -euo pipefail
MEOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MEOW_ROOT"
DATA_ROOT="${1:?Provide the extracted laser_benchmark/v1 directory}"
CHECKPOINT="${2:?Provide the MEOW-sB25.pth checkpoint}"
OUTPUT_ROOT="${3:?Provide an output directory}"
shift 3
if [ "$#" -eq 0 ]; then
    set -- blk_mixed blk_erp blk_pinhole blk_fisheye
fi
for TRACK in "$@"; do
    DATA="$DATA_ROOT/$TRACK"
    OUTPUT="$OUTPUT_ROOT/$TRACK"
    mkdir -p "$OUTPUT"
    bash scripts/run_npu.sh scripts/realset/predict_ours.py \
        --tuples "$DATA/tuples.json" --frames-root "$DATA" \
        --ckpt "MEOW=$CHECKPOINT:wrap" --input-resize squeeze --pano auto \
        --device npu --out "$OUTPUT/preds" 2>&1 | tee "$OUTPUT/predict.log"
    bash scripts/run_npu.sh scripts/realset/eval_realset.py \
        --tuples "$DATA/tuples.json" --preds-root "$OUTPUT/preds" \
        --models MEOW --out "$OUTPUT/eval" --scoring-device cpu \
        2>&1 | tee "$OUTPUT/eval.log"
    bash scripts/run_npu.sh - "$DATA/tuples.json" "$OUTPUT/eval/results_MEOW.json" <<'PY'
import json
import math
import sys
spec = json.load(open(sys.argv[1]))
result = json.load(open(sys.argv[2]))
a = result['aggregate']
n = len(spec['tuples'])
assert a['n_cases'] == n and a['n_missing_preds'] == 0, a
assert a['n_pointmap'] == n and a['n_pointmap_failed'] == 0, a
for row in result['per_tuple']:
    assert all(math.isfinite(row[k]) for k in ('rra30', 'rta30', 'auc30', 'acc', 'comp', 'nc')), row
assert {row['id'] for row in result['per_tuple']} == {row['id'] for row in spec['tuples']}
print(f'PASS: all {n} tuples have finite pose and pointmap metrics')
PY
done
