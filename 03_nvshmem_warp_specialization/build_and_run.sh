#!/usr/bin/env bash
# 任务 03：编译并运行 NVSHMEM warp specialization 示例。
# 用法: ./build_and_run.sh [N] [COMPUTE_ITERS]   （参数透传给二进制）
# 环境变量: NVSHMEM_HOME 必填；NPES 可选（进程数，默认 2）；
#           NVCC 可选（默认 nvcc）。
# 环境要求见本目录 README.md（>=2 张 P2P GPU + nvcc + NVSHMEM）。
set -euo pipefail
cd "$(dirname "$0")"

: "${NVSHMEM_HOME:?请先 export NVSHMEM_HOME=/path/to/nvshmem（见本目录 README.md）}"
NVCC="${NVCC:-nvcc}"
NPES="${NPES:-2}"

mkdir -p build
"$NVCC" -O2 -std=c++17 -rdc=true \
    -I"${NVSHMEM_HOME}/include" 03_nvshmem_warp_specialization.cu \
    -L"${NVSHMEM_HOME}/lib" -lnvshmem_host -lnvshmem_device \
    -o build/nvshmem_warp_specialization
"${NVSHMEM_HOME}/bin/nvshmrun" -np "${NPES}" ./build/nvshmem_warp_specialization "$@"
