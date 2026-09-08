#!/usr/bin/env bash
# 任务 07:编译并运行 block slot 占用微基准(Python 基准部分见 README.md)。
# 用法: ./build_and_run.sh [CASE]
#   CASE = all | small | wide | smem  (默认 all)
#     small: 32 threads/block,无 dynamic smem——blocks 先撞 CTA/block slot 上限
#     wide : 256 threads/block——单 block 占满 warp slot,总驻留 block 数下降
#     smem : 128 threads/block + 48KB dynamic smem——改由 shared memory 限制驻留
# 环境要求见本目录 README.md(>=1 张 GPU + nvcc,单 GPU 即可)。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"

mkdir -p build
"$NVCC" -O3 -lineinfo -Wno-deprecated-gpu-targets \
  block_slot_demo.cu -o build/block_slot_demo

case "${1:-all}" in
  small) ./build/block_slot_demo 32 0 4096 200000 ;;
  wide)  ./build/block_slot_demo 256 0 4096 200000 ;;
  smem)  ./build/block_slot_demo 128 49152 4096 200000 ;;
  all)
    ./build/block_slot_demo 32 0 4096 200000
    ./build/block_slot_demo 256 0 4096 200000
    ./build/block_slot_demo 128 49152 4096 200000
    ;;
  *)
    echo "Usage: $0 {all|small|wide|smem}" >&2
    exit 2
    ;;
esac
