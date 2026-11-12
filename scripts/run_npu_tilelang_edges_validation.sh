#!/usr/bin/env bash
# ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang_edges_validation.sh
set -euo pipefail
MEOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MEOW_ROOT"
bash scripts/run_npu_tilelang.sh scripts/npu_validate_tilelang_edges.py --trace
bash scripts/run_npu_tilelang.sh scripts/npu_profile_tilelang_edges.py
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_postprocess.py \
    --fused-edges --out work/npu-edges/postprocess --trace
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_model.py \
    --stage full --runner --fused-edges --compare-unfused \
    --tracks blk_mixed blk_erp blk_pinhole blk_fisheye --out work/npu-edges/full --trace
bash scripts/run_npu.sh scripts/summarize_npu_graph.py work/npu-edges
