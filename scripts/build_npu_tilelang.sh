#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
source "${MEOW_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
export PATH="$repo/.venv/bin:$PATH"
export LD_PRELOAD=/lib64/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}
src="$repo/work/npu-k1/tilelang/source"
# Source checkout is pinned in the results report; SDK and system compiler are unchanged.
bash "$src/3rdparty/patches/apply_tvm_patches.sh"
cmake -S "$src" -B "$src/build" -G 'Unix Makefiles' \
  -DUSE_ASCEND=ON -DUSE_LLVM=OFF -DUSE_CUDA=OFF -DUSE_GTEST=OFF \
  -DCMAKE_C_COMPILER="$repo/work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$repo/work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-g++"
cmake --build "$src/build" -j "${MEOW_BUILD_JOBS:-16}"
