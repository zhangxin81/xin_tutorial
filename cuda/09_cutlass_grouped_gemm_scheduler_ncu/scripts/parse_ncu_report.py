#!/usr/bin/env python3
"""Parse the raw NCU CSVs exported by profile_ncu.sh.

Prints one compact block per kernel: kernel-name decode (schedule family,
tile shape, mainloop stages), duration, grid/waves, SM/tensor-pipe
throughput. Works offline on the exported CSVs, no GPU required. Defaults to
../../worker_results/09_cutlass_grouped_gemm_scheduler_ncu/raw_csv/*.csv
relative to this script; pass explicit CSV paths to override.
"""

import csv
import glob
import os
import re
import sys


def kernel_decode(name):
    info = {}
    m = re.search(r"KernelPtrArrayTmaWarpSpecialized(\w+)", name)
    info["schedule"] = m.group(1) if m else "?"
    m = re.search(r"WarpSpecialized\w+>, tuple<C<(\d+)>, C<(\d+)>, C<(\d+)>>", name)
    info["tile_mnk"] = "x".join(m.groups()) if m else "?"
    m = re.search(r"MainloopSm90ArrayTmaGmmaWarpSpecialized<(\d+)", name)
    info["mainloop_stages"] = m.group(1) if m else "?"
    m = re.search(r"MMA_(\d+x\d+x\d+)_\w+_(\w+)", name)
    info["wgmma_atom"] = f"{m.group(1)} src={m.group(2)}" if m else "?"
    info["tma_multicast"] = "SM90_TMA_LOAD_MULTICAST" in name
    return info


def main(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            rows.append((path, list(csv.DictReader(fh))))

    print(f"{'path':44s} {'schedule':12s} {'tile':10s} "
          f"{'dur_us':>8s} {'grid':>5s} {'waves':>6s} {'sm%':>6s} {'tensor%':>8s}")
    for path, rows_ in rows:
        by_name = {r["Metric Name"]: r["Metric Value"] for r in rows_}
        kernel = rows_[0]["Kernel Name"]
        d = kernel_decode(kernel)
        print(f"{path.split('/')[-1]:44s} {d['schedule']:12s} {d['tile_mnk']:10s} "
              f"{by_name.get('gpu__time_duration.avg', '?'):>8s} "
              f"{by_name.get('launch__grid_dim_x', '?'):>5s} "
              f"{by_name.get('launch__waves_per_multiprocessor', '?'):>6s} "
              f"{by_name.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', '?'):>6s} "
              f"{by_name.get('sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active', '?'):>8s}")
        print(f"{'':44s} stages={d['mainloop_stages']} wgmma={d['wgmma_atom']} "
              f"multicast={d['tma_multicast']}")


if __name__ == "__main__":
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "..", "worker_results", "09_cutlass_grouped_gemm_scheduler_ncu", "raw_csv")
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(default_dir, "*.csv")))
    if not paths:
        sys.exit(f"no CSVs given and none found under {default_dir}")
    main(paths)
