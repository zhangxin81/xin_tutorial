#!/usr/bin/env bash
# Task 10 one-shot script: compile the root CUDA candidate and run a verified
# smoke benchmark (correctness on all 16 official workloads + score estimate).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

python3 experiments/score_aligned_benchmark.py \
  --configs official,triton01,root_current \
  --rounds 3 --repeats 50

echo
echo "Smoke run OK. For the full-protocol benchmark and the NCU workflow, see README.md."
