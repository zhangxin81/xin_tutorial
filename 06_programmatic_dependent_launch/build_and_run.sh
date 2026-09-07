#!/usr/bin/env bash
# 任务 06:编译并运行 PDL(Programmatic Dependent Launch) producer/consumer 示例。
# 用法: ./build_and_run.sh [demo] [额外参数...]
#   demo = rmsnorm | benefit | triton | all (默认 all)
# 环境要求见本目录 README.md(≥1 张 SM90+ GPU;triton 子命令还需 torch/triton)。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-sm_90a}"
DEMO="${1:-all}"

mkdir -p build

if [[ "$DEMO" == "rmsnorm" || "$DEMO" == "all" ]]; then
  "$NVCC" -O3 -std=c++17 -Wno-deprecated-gpu-targets -arch="$ARCH" \
    pdl_rmsnorm_qkv.cu -o build/pdl_rmsnorm_qkv
  for cycles in 0 50000 100000 300000 1000000; do
    ./build/pdl_rmsnorm_qkv --mode both --m 64 --k 4096 --n 768 \
      --warmup 10 --iters 50 --tail-cycles "$cycles"
  done
fi

if [[ "$DEMO" == "benefit" || "$DEMO" == "all" ]]; then
  "$NVCC" -O3 -std=c++17 -Wno-deprecated-gpu-targets -arch="$ARCH" \
    pdl_overlap_benefit.cu -o build/pdl_overlap_benefit
  for tail in 0 500000 1000000; do
    for prologue in 0 500000 1000000; do
      ./build/pdl_overlap_benefit --mode both \
        --producer-tail-cycles "$tail" --consumer-prologue-cycles "$prologue" \
        --consumer-body-cycles 100000 --warmup 5 --iters 50
    done
  done
fi

if [[ "$DEMO" == "triton" || "$DEMO" == "all" ]]; then
  for iters in 32 128 512; do
    python pdl_rmsnorm_qkv_triton.py --mode both --m 64 --k 4096 --n 768 \
      --tail-iters "$iters" --warmup 10 --iters 50
  done
fi
