#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TMA_PATTERNS = [
    re.compile(r"\butmaldg\b", re.IGNORECASE),
    re.compile(r"\butmastg\b", re.IGNORECASE),
    re.compile(r"\butmacmdflush\b", re.IGNORECASE),
    re.compile(r"\bcp\.async\.bulk\b", re.IGNORECASE),
    re.compile(r"\bcp\.reduce\.async\.bulk\b", re.IGNORECASE),
    re.compile(r"\btma\b", re.IGNORECASE),
]
TENSOR_CORE_PATTERNS = [
    re.compile(r"\bhgmma\b", re.IGNORECASE),
    re.compile(r"\bwgmma\b", re.IGNORECASE),
    re.compile(r"\bwgmma\.mma_async\b", re.IGNORECASE),
    re.compile(r"\bmma\.sync\b", re.IGNORECASE),
    re.compile(r"\bhmma\b", re.IGNORECASE),
]


def read_long_metrics(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row.get("Metric Name") or row.get("Metric Name ")
            if name:
                rows.append(row)
    return rows


def read_wide_metrics(path: Path) -> tuple[list[str], dict[str, str], list[dict[str, str]]]:
    if not path.exists():
        return [], [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], [], []
    header = rows[0]
    unit_row = rows[1] if len(rows) > 1 else []
    units = {header[i]: unit_row[i] if i < len(unit_row) else "" for i in range(len(header))}
    data_rows = []
    for raw in rows[2:]:
        if any(cell.strip() for cell in raw):
            data_rows.append({header[i]: raw[i] if i < len(raw) else "" for i in range(len(header))})
    return header, units, data_rows


def metric_value(row: dict[str, str]) -> str:
    for key in ("Metric Value", "Avg", "Value", "Max", "Min"):
        if key in row and row[key]:
            return row[key]
    return ""


def find_lines(path: Path, patterns: list[re.Pattern[str]]) -> list[str]:
    if not path.exists():
        return []
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(p.search(line) for p in patterns):
            stripped = line.strip()
            if stripped and stripped not in matches:
                matches.append(stripped)
    return matches


def is_nonzero(value: str) -> bool:
    try:
        return float(value.replace(",", "")) != 0.0
    except ValueError:
        return bool(value.strip())


def selected_wide_metrics(
    header: list[str], units: dict[str, str], rows: list[dict[str, str]]
) -> list[tuple[str, str, str, str]]:
    wanted_exact = [
        "Block Size",
        "Grid Size",
        "Device",
        "CC",
        "device__attribute_tensor_map_access_supported",
        "gpu__time_duration.avg",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum",
        "l1tex__m_l1tex2xbar_write_sectors_mem_dshared_op_tma_st.sum",
        "l1tex__m_l1tex2xbar_write_sectors_mem_dshared_op_tma_red.sum",
        "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hmma_pred_on.avg",
        "sm__inst_executed_pipe_tensor_op_hmma.sum",
    ]
    wanted_substrings = (
        "hgmma_src_bf16_dst_fp32",
        "tensor_op",
        "op_hmma",
        "op_wgmma",
        "op_mma",
        "_op_tma_",
    )
    skip_substrings = (
        "peak_sustained",
        "bgmma",
        "bmma",
        "dmma",
        "imma",
        "f16_dst",
        "f64_dst",
        "s8_dst",
    )
    metrics: list[tuple[str, str, str]] = []
    for row in rows[:4]:
        kernel = row.get("Kernel Name", "")
        seen = set()
        for name in wanted_exact:
            value = row.get(name, "")
            if value:
                metrics.append((name, kernel, value, units.get(name, "")))
                seen.add(name)
        for name in header:
            lname = name.lower()
            if name in seen or not any(token in lname for token in wanted_substrings):
                continue
            if any(token in lname for token in skip_substrings):
                continue
            value = row.get(name, "")
            if value and is_nonzero(value):
                metrics.append((name, kernel, value, units.get(name, "")))
                seen.add(name)
            if len(metrics) >= 48:
                break
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    long_rows = read_long_metrics(args.raw_csv)
    wide_header, wide_units, wide_rows = read_wide_metrics(args.raw_csv)
    interesting_metrics = []
    wanted = (
        "tensor",
        "hmma",
        "wgmma",
        "sass",
        "dram",
        "throughput",
        "duration",
        "l1tex",
        "l2",
    )
    for row in long_rows:
        name = row.get("Metric Name", "")
        if any(token in name.lower() for token in wanted):
            interesting_metrics.append((name, row.get("Kernel Name", ""), metric_value(row), ""))
    if not interesting_metrics:
        interesting_metrics = selected_wide_metrics(wide_header, wide_units, wide_rows)

    tma_lines = find_lines(args.sass, TMA_PATTERNS)
    tensor_core_lines = find_lines(args.sass, TENSOR_CORE_PATTERNS)

    out = []
    out.append("# Nsight Compute Summary")
    out.append("")
    out.append("## Instruction Evidence")
    out.append("")
    out.append(f"- TMA-related SASS lines found: {len(tma_lines)}")
    out.append(f"- Tensor Core-related SASS lines found: {len(tensor_core_lines)}")
    out.append("")
    out.append("### TMA Lines")
    if tma_lines:
        out.extend(f"- `{line}`" for line in tma_lines[:40])
    else:
        out.append("- None found in exported SASS. For H100, expected mnemonics include `UTMALDG`, `UTMASTG`, `UTMACMDFLUSH`, or PTX-style `cp.async.bulk.tensor`.")
    out.append("")
    out.append("### Tensor Core Lines")
    if tensor_core_lines:
        out.extend(f"- `{line}`" for line in tensor_core_lines[:40])
    else:
        out.append("- None found in exported SASS. For H100 BF16 GEMM, expected mnemonics include `HGMMA`, `WGMMA`, `mma.sync`, or HMMA-family tensor instructions.")
    out.append("")
    out.append("## Selected Metrics")
    out.append("")
    if interesting_metrics:
        out.append("| Kernel | Metric | Value | Unit |")
        out.append("|---|---|---:|---|")
        for name, kernel, value, unit in interesting_metrics[:48]:
            out.append(f"| `{kernel}` | `{name}` | `{value}` | `{unit}` |")
    else:
        out.append("- No matching metrics were parsed from raw CSV.")
    out.append("")
    out.append("## How To Read This")
    out.append("")
    out.append("- TMA evidence normally appears as SM90 bulk asynchronous copy instructions in the Source/SASS page, including Hopper SASS mnemonics such as `UTMALDG` and `UTMASTG`.")
    out.append("- Tensor Core evidence appears as HGMMA/WGMMA/HMMA/MMA instructions and non-zero tensor-op instruction metrics.")
    out.append("- The throughput and memory metrics provide the surrounding performance context.")

    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
