#!/usr/bin/env bash
# Profile the root implementation (rows=8192 and rows=131) with Nsight Compute.
#
# NCU_BIN: path to the ncu CLI. Defaults to `ncu` from PATH; the original
# captures used Nsight Compute 2025.2.1 on a B200.
#
# --launch-skip skips the warmup launches of the same kernel that match the
# filter, so the capture lands on a steady-state launch. Outputs (including
# the .ncu-rep and raw exports) go to experiments/ncu_out/, which is
# git-ignored; nothing under experiments/ depends on them being present.
set -euo pipefail

cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export NO_COLOR=1
export TERM=dumb

NCU_BIN="${NCU_BIN:-ncu}"
OUT_DIR="experiments/ncu_out"
mkdir -p "${OUT_DIR}"

for shape in 1,8192 1,131; do
  label="${shape/,/x}"
  report="${OUT_DIR}/root_current_${label}"
  "${NCU_BIN}" --force-overwrite --target-processes all --set full \
    --kernel-name regex:fused_add_rmsnorm \
    --launch-skip 60 --launch-count 1 \
    --export "${report}" \
    python3 experiments/profile_candidate.py --config root_current --shape "${shape}" \
      --warmup 50 --iters 80 --flush-l2 \
    > "${report}.stdout.txt" 2>&1
  "${NCU_BIN}" --import "${report}.ncu-rep" --page raw --csv > "${report}.raw.csv"
  "${NCU_BIN}" --import "${report}.ncu-rep" --page source --print-source sass > "${report}.sass.txt"
done
