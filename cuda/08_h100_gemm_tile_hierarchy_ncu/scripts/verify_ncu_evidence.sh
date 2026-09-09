#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT_DIR}/reports"
SUMMARY="${REPORT_DIR}/ncu_summary.md"
SASS="${REPORT_DIR}/sass.txt"
RAW="${REPORT_DIR}/raw.csv"
REPORT="${REPORT_DIR}/h100_cublaslt_gemm.ncu-rep"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

test -s "${REPORT}" || fail "missing non-empty .ncu-rep report"
test -s "${RAW}" || fail "missing non-empty raw.csv"
test -s "${SASS}" || fail "missing non-empty sass.txt"
test -s "${SUMMARY}" || fail "missing non-empty ncu_summary.md"

search_file() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${pattern}" "${file}"
  else
    grep -Eiq "${pattern}" "${file}"
  fi
}

extract_count() {
  local label="$1"
  awk -F': ' -v label="$label" '$0 ~ label {print $2; exit}' "${SUMMARY}" | tr -dc '0-9'
}

tma_count="$(extract_count "TMA-related SASS lines found")"
tensor_count="$(extract_count "Tensor Core-related SASS lines found")"

if [[ -z "${tensor_count}" || "${tensor_count}" -le 0 ]]; then
  fail "summary has no parsed Tensor Core SASS evidence"
fi

if [[ -z "${tma_count}" || "${tma_count}" -le 0 ]]; then
  fail "summary has no parsed TMA SASS evidence"
fi

if ! search_file "HGMMA|WGMMA|HMMA|MMA\\.SYNC" "${SASS}"; then
  fail "Tensor Core instruction mnemonic not found in SASS"
fi

if ! search_file "UTMALDG|UTMASTG|UTMACMDFLUSH|CP\\.ASYNC\\.BULK" "${SASS}"; then
  fail "TMA instruction mnemonic not found in SASS"
fi

echo "NCU evidence audit passed."
