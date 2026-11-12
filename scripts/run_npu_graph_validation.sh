#!/usr/bin/env bash
# Usage: ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_graph_validation.sh
# Uses the existing checkpoint, benchmark images and postprocess raw caches.
set -euo pipefail
MEOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MEOW_ROOT"
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_k1.py
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_postprocess.py --out work/npu-graph/postprocess-batched --trace
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_model.py --stage core --trace
bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_model.py --stage full --runner \
  --tracks blk_mixed blk_erp blk_pinhole blk_fisheye --out work/npu-graph/full --trace
bash scripts/run_npu.sh scripts/summarize_npu_graph.py work/npu-graph
