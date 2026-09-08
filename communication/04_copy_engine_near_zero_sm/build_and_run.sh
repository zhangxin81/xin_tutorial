#!/usr/bin/env bash
# 任务 04：编译并运行 copy engine near-zero-SM 示例。
# 用法: ./build_and_run.sh [MIB]   （参数透传给二进制，默认 128）
# 环境要求见本目录 README.md（>=2 张支持 P2P 的 GPU + nvcc）。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"

mkdir -p build
"$NVCC" -O2 -std=c++17 04_copy_engine_near_zero_sm.cu -o build/copy_engine_near_zero_sm
./build/copy_engine_near_zero_sm "$@"
