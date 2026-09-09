#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
mkdir -p "${BUILD_DIR}"

if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
    if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
      export PATH="${cuda_dir}/bin:${PATH}"
      export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi

NVCC="${NVCC:-nvcc}"
"${NVCC}" -O3 -std=c++17 -arch=sm_90 \
  "${ROOT_DIR}/src/h100_cublaslt_gemm.cu" \
  -lcublasLt -lcublas \
  -o "${BUILD_DIR}/h100_cublaslt_gemm"

echo "${BUILD_DIR}/h100_cublaslt_gemm"
