#!/usr/bin/env bash
# Profile the case-1 trio (torch grouped_mm, CUTLASS cooperative, CUTLASS
# pingpong) with Nsight Compute. Profiling output stays outside the repo, in
# ../../worker_results/09_cutlass_grouped_gemm_scheduler_ncu/ (override with
# WORKER_RESULTS):
#   <out>/<name>.ncu-rep       binary report (open with the NCU UI)
#   <out>/raw_csv/<name>.csv   raw metric table (ncu --csv)
#   <out>/sass_<name>.txt      SASS export (input for a sass excerpt)
# Set FULL=1 to collect --set full instead of the five headline metrics
# (many replay passes; that is what produced the full-section captures).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${WORKER_RESULTS:-${ROOT_DIR}/../../worker_results/09_cutlass_grouped_gemm_scheduler_ncu}"
CSV_DIR="${OUT_DIR}/raw_csv"
BIN="${ROOT_DIR}/build/cutlass_grouped_gemm"
CASE_ARGS=(--groups=10 --m=128 --n=2048 --k=1024 --alpha=1 --beta=0 --no-verify)
METRICS="gpu__time_duration.avg,launch__grid_dim_x,launch__grid_dim_y,launch__grid_dim_z,launch__waves_per_multiprocessor,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
mkdir -p "${OUT_DIR}" "${CSV_DIR}"

for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
  if [[ -d "${cuda_dir}/bin" ]]; then
    export PATH="${cuda_dir}/bin:${PATH}"
  fi
done

NCU_BIN="${NCU_BIN:-$(command -v ncu || true)}"
if [[ -z "${NCU_BIN}" ]]; then
  for candidate in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu \
                   /opt/nvidia/nsight-compute/*/ncu; do
    if [[ -x "${candidate}" ]]; then NCU_BIN="${candidate}"; break; fi
  done
fi
if [[ -z "${NCU_BIN}" ]]; then
  echo "ERROR: ncu not found; set NCU_BIN=/path/to/ncu" >&2
  exit 1
fi

COLLECT=(--launch-skip 1 --launch-count 1 --force-overwrite)
if [[ "${FULL:-0}" == "1" ]]; then
  COLLECT+=(--set full)
else
  COLLECT+=(--metrics "${METRICS}")
fi

profile() {
  local name="$1"; shift
  echo "== profiling ${name} =="
  "${NCU_BIN}" "${COLLECT[@]}" --export "${OUT_DIR}/${name}" "$@" 2> "${OUT_DIR}/${name}_stdout.txt"
  "${NCU_BIN}" --import "${OUT_DIR}/${name}.ncu-rep" --csv > "${CSV_DIR}/${name}.csv"
  "${NCU_BIN}" --import "${REPORT_DIR}/${name}.ncu-rep" --page source --print-source sass \
    > "${OUT_DIR}/sass_${name}.txt" 2>/dev/null || true
}

profile torch_pingpong_64x128x128 \
  python3 "${ROOT_DIR}/src/profile_grouped_mm.py" --warmup 5 --repeat 1
profile cutlass_cooperative_128x256x64 \
  "${BIN}" --schedule=coop "${CASE_ARGS[@]}" --iterations=20
profile cutlass_pingpong_64x128x128 \
  "${BIN}" --schedule=pingpong "${CASE_ARGS[@]}" --iterations=20

python3 "${ROOT_DIR}/scripts/parse_ncu_report.py" "${CSV_DIR}"
echo "Done. Output under ${OUT_DIR}"
