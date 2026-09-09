#!/usr/bin/env bash
# Task 08 one-shot script: build the sm_90 binary and run a small verified GEMM.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/h100_cublaslt_gemm"

"${ROOT_DIR}/scripts/build.sh"

"${BIN}" --m=4096 --n=4096 --k=4096 --warmup=5 --iters=10 --verify

echo
echo "Smoke test OK. For the full 8192 benchmark and the NCU workflow, see README.md."
