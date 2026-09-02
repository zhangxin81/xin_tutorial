#!/usr/bin/env bash
# 任务 05:编译并运行 CUDA Graph capture/replay 易错场景示例。
# 用法: ./build_and_run.sh [SCENARIO]
#   SCENARIO = all | perf | s1bad | s1fix_data | s1fix_update | s2bad | s2fix | s3bad | s3fix
#   (默认 all;参数透传给二进制)
# 环境要求见本目录 README.md(>=1 张 GPU + nvcc,单 GPU 即可)。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"

mkdir -p build
"$NVCC" -O2 -std=c++17 -Wno-deprecated-gpu-targets \
  05_cuda_graph_pitfalls.cu -o build/cuda_graph_pitfalls
./build/cuda_graph_pitfalls "${1:-all}"
