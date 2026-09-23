#!/usr/bin/env bash
# Build the grouped-GEMM harness (based on NVIDIA CUTLASS example 57).
# Requires CUTLASS >= 3.x headers; point CUTLASS_DIR at a checkout if it is
# not already on the include path.
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

if [[ -z "${CUTLASS_DIR:-}" ]]; then
  echo "ERROR: set CUTLASS_DIR to a CUTLASS >= 3.x checkout (git clone https://github.com/nvidia/cutlass)" >&2
  exit 1
fi

NVCC="${NVCC:-nvcc}"
# sm_90a is required: the kernel uses TMA descriptor modification on Hopper
# (CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED).
"${NVCC}" -O3 -std=c++17 -arch=sm_90a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  -I"${CUTLASS_DIR}/examples/common" \
  "${ROOT_DIR}/src/cutlass_grouped_gemm.cu" \
  -o "${BUILD_DIR}/cutlass_grouped_gemm"

echo "${BUILD_DIR}/cutlass_grouped_gemm"
