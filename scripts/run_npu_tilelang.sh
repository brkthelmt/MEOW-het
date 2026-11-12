#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
export TL_ROOT="$PWD/work/npu-k1/tilelang/source"
export PYTHONPATH="$TL_ROOT:$TL_ROOT/3rdparty/tvm/python${PYTHONPATH:+:$PYTHONPATH}"
export ACL_OP_INIT_MODE=1
export TILELANG_ASCEND_DEVICE_NAME=Ascend910B2
export CXX="$PWD/work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-g++"
export CC="$PWD/work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-gcc"
exec bash scripts/run_npu.sh "$@"
