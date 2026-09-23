#!/usr/bin/env bash
# Task 09 one-shot script: build the grouped-GEMM harness and smoke-run both
# schedules on the case shape (groups=10, m=128, k=1024, n=2048).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/cutlass_grouped_gemm"

"${ROOT_DIR}/scripts/build.sh"

COMMON=(--groups=10 --m=128 --n=2048 --k=1024 --alpha=1 --beta=0 --iterations=1000 --no-verify)

echo "== cooperative schedule =="
"${BIN}" --schedule=coop "${COMMON[@]}"
echo
echo "== pingpong schedule =="
"${BIN}" --schedule=pingpong "${COMMON[@]}"

echo
echo "Smoke test OK. For the torch comparison and the NCU workflow, see README.md."
