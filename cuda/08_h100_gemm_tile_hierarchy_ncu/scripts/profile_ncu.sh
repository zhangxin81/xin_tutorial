#!/usr/bin/env bash
# Profile one cuBLASLt GEMM kernel with Nsight Compute, then export evidence:
#   reports/h100_cublaslt_gemm.ncu-rep  binary report (open with NCU UI)
#   reports/raw.csv                    raw metric table
#   reports/sass.txt                   SASS with per-instruction counters
#   reports/ncu_summary.md             parsed TMA / Tensor Core evidence
# Each run also copies the exports under reports/archive/<UTC tag>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/h100_cublaslt_gemm"
REPORT_DIR="${ROOT_DIR}/reports"
REPORT_BASE="${REPORT_DIR}/h100_cublaslt_gemm"
RUN_TAG="${RUN_TAG:-$(date -u '+%Y%m%dT%H%M%SZ')}"
ARCHIVE_DIR="${REPORT_DIR}/archive/${RUN_TAG}"
NCU_BIN="${NCU_BIN:-}"
NCU_METRICS="${NCU_METRICS:-}"
mkdir -p "${REPORT_DIR}" "${ARCHIVE_DIR}"

for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
  if [[ -d "${cuda_dir}/bin" ]]; then
    export PATH="${cuda_dir}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
  fi
done

if [[ ! -x "${BIN}" ]]; then
  "${ROOT_DIR}/scripts/build.sh"
fi

if [[ -z "${NCU_BIN}" ]]; then
  NCU_BIN="$(command -v ncu || true)"
fi
if [[ -z "${NCU_BIN}" ]]; then
  for candidate in \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-*/bin/ncu \
    /opt/nvidia/nsight-compute/*/ncu \
    /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_*/ncu; do
    if [[ -x "${candidate}" ]]; then
      NCU_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${NCU_BIN}" || ! -x "${NCU_BIN}" ]]; then
  cat >&2 <<'MSG'
ncu was not found in PATH.

Nsight Compute is usually bundled with the CUDA toolkit under:
  /usr/local/cuda/bin/ncu
  /usr/local/cuda-*/bin/ncu
  /opt/nvidia/nsight-compute/*/ncu

If it is not installed, download the Linux Nsight Compute CLI from
NVIDIA Developer, or use a CUDA image/package that already includes
ncu, then prepend its bin directory to PATH or set NCU_BIN.
MSG
  exit 127
fi

if ! nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee "${REPORT_DIR}/gpu.txt" | grep -q 'H100'; then
  echo "This tutorial is intended for H100. Continuing, but TMA/WGMMA evidence may differ." >&2
fi

"${NCU_BIN}" --version | tee "${REPORT_DIR}/ncu_version.txt"

for existing in \
  "${REPORT_BASE}.ncu-rep" \
  "${REPORT_DIR}/raw.csv" \
  "${REPORT_DIR}/sass.txt" \
  "${REPORT_DIR}/ncu_summary.md" \
  "${REPORT_DIR}/profile_stdout.txt" \
  "${REPORT_DIR}/gpu.txt" \
  "${REPORT_DIR}/ncu_version.txt"; do
  if [[ -f "${existing}" ]]; then
    cp -p "${existing}" "${ARCHIVE_DIR}/previous_$(basename "${existing}")"
  fi
done

NCU_ARGS=(
  --force-overwrite
  --target-processes all
  --launch-skip 20
  --launch-count 1
  --set full
  --export "${REPORT_BASE}"
)
if [[ -n "${NCU_METRICS}" ]]; then
  NCU_ARGS+=(--metrics "${NCU_METRICS}")
fi

"${NCU_BIN}" "${NCU_ARGS[@]}" "${BIN}" "$@" | tee "${REPORT_DIR}/profile_stdout.txt"

"${NCU_BIN}" --import "${REPORT_BASE}.ncu-rep" --page raw --csv > "${REPORT_DIR}/raw.csv"
"${NCU_BIN}" --import "${REPORT_BASE}.ncu-rep" --page source --print-source sass > "${REPORT_DIR}/sass.txt" || true

python3 "${ROOT_DIR}/scripts/parse_ncu_report.py" \
  --raw-csv "${REPORT_DIR}/raw.csv" \
  --sass "${REPORT_DIR}/sass.txt" \
  --out "${REPORT_DIR}/ncu_summary.md"

cp -p "${REPORT_BASE}.ncu-rep" "${ARCHIVE_DIR}/h100_cublaslt_gemm.ncu-rep"
cp -p "${REPORT_DIR}/raw.csv" "${ARCHIVE_DIR}/raw.csv"
cp -p "${REPORT_DIR}/sass.txt" "${ARCHIVE_DIR}/sass.txt"
cp -p "${REPORT_DIR}/ncu_summary.md" "${ARCHIVE_DIR}/ncu_summary.md"
cp -p "${REPORT_DIR}/profile_stdout.txt" "${ARCHIVE_DIR}/profile_stdout.txt"
cp -p "${REPORT_DIR}/gpu.txt" "${ARCHIVE_DIR}/gpu.txt"
cp -p "${REPORT_DIR}/ncu_version.txt" "${ARCHIVE_DIR}/ncu_version.txt"

echo "Report: ${REPORT_BASE}.ncu-rep"
echo "Summary: ${REPORT_DIR}/ncu_summary.md"
echo "Archived NCU exports: ${ARCHIVE_DIR}"
