#!/usr/bin/env bash
# Usage: ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu.sh scripts/npu_smoke.py
set -e
MEOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${MEOW_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
export MEOW_DEVICE="${MEOW_DEVICE:-npu}"
export MEOW_DINOV2_SOURCE=bundled
export XFORMERS_DISABLED=1
export PYTHONPATH="$MEOW_ROOT:$MEOW_ROOT/third_party/uniception${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
# CANN's msprof launches python3 from PATH. Use the same environment as the
# model, rather than the host's Python 3.7 and its incompatible libffi.
MEOW_PYTHON="${MEOW_PYTHON:-$MEOW_ROOT/.venv/bin/python}"
export PATH="$(dirname "$MEOW_PYTHON"):$PATH"
cd "$MEOW_ROOT"
exec "$MEOW_PYTHON" "$@"
