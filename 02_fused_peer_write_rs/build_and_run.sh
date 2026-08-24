#!/usr/bin/env bash
# 任务 02：编译并运行 GEMM+RS peer write 示例。
# 用法: ./build_and_run.sh [M] [N] [K]   （参数透传给二进制，默认 512 512 512）
# 环境要求见本目录 README.md（>=2 张支持 P2P 的 GPU + nvcc）。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"

mkdir -p build
"$NVCC" -O2 -std=c++17 02_fused_peer_write_rs.cu -o build/fused_peer_write_rs
./build/fused_peer_write_rs "$@"
