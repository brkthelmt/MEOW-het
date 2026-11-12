#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source "${MEOW_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
PY="${MEOW_PYTHON:-$PWD/.venv/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
set -u
# Build-only fix for system make/guile vs the inherited Conda libffi.
export LD_PRELOAD=/lib64/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}
"$PY" operators/k1/prepare_cmake.py /usr/local/Ascend/cann-9.1.0/aarch64-linux/tikcpp/ascendc_kernel_cmake "$PWD/work/npu-k1/cann-cmake"
# CANN 9.1 preprocess merging rewrites object files; use a clean generated tree.
"$PY" -c 'import shutil; shutil.rmtree("work/npu-k1/build-gcc13", ignore_errors=True)'
cmake -DMEOW_ASCENDC_CMAKE="$PWD/work/npu-k1/cann-cmake/ascendc.cmake" -S operators/k1 -B work/npu-k1/build-gcc13 -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=$PWD/work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-g++ -DPython3_EXECUTABLE="$PY" -DASCEND_PYTHON_EXECUTABLE="$PY" -DCMAKE_BUILD_TYPE=Release
cmake --build work/npu-k1/build-gcc13 -j 4
