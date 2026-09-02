#!/usr/bin/env python3
"""Summarize overlap-related evidence from Nsight Systems sqlite exports."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _fetch_kernel_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT k.start, k.end, k.deviceId, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """
    return [
        {
            "start_ns": int(row[0]),
            "end_ns": int(row[1]),
            "device": row[2],
            "stream": row[3],
            "name": row[4] or "",
            "duration_us": (int(row[1]) - int(row[0])) / 1000.0,
        }
        for row in conn.execute(query)
    ]


def _fetch_memcpy_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'"
    ).fetchone()
    if not exists:
        return []
    query = """
        SELECT start, end, deviceId, streamId, bytes, srcDeviceId, dstDeviceId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        ORDER BY start
    """
    return [
        {
            "start_ns": int(row[0]),
            "end_ns": int(row[1]),
            "device": row[2],
            "stream": row[3],
            "bytes": int(row[4]),
            "src_device": row[5],
            "dst_device": row[6],
            "duration_us": (int(row[1]) - int(row[0])) / 1000.0,
        }
        for row in conn.execute(query)
    ]


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _merged_duration_us(intervals: list[tuple[int, int]]) -> float:
    merged = _merge_intervals(intervals)
    return sum(end - start for start, end in merged) / 1000.0


def _intersection_us(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    total = 0
    i = 0
    j = 0
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    while i < len(left) and j < len(right):
        ls, le = left[i]
        rs, re = right[j]
        total += max(0, min(le, re) - max(ls, rs))
        if le < re:
            i += 1
        else:
            j += 1
    return total / 1000.0


def _summarize_pair(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_intervals = [(item["start_ns"], item["end_ns"]) for item in left]
    right_intervals = [(item["start_ns"], item["end_ns"]) for item in right]
    overlap_us = _intersection_us(left_intervals, right_intervals)
    all_intervals = left_intervals + right_intervals
    span_us = (
        (max(end for _, end in all_intervals) - min(start for start, _ in all_intervals))
        / 1000.0
        if all_intervals
        else 0.0
    )
    return {
        f"{left_label}_count": len(left),
        f"{right_label}_count": len(right),
        f"{left_label}_us": _merged_duration_us(left_intervals),
        f"{right_label}_us": _merged_duration_us(right_intervals),
        "overlap_us": overlap_us,
        "active_span_us": span_us,
        "overlap_pct_of_active_span": 100.0 * overlap_us / span_us if span_us > 0 else 0.0,
        f"{left_label}_events": left,
        f"{right_label}_events": right,
    }


def analyze(sqlite_path: Path, task: str) -> dict[str, Any]:
    with sqlite3.connect(sqlite_path) as conn:
        kernels = _fetch_kernel_rows(conn)
        memcpys = _fetch_memcpy_rows(conn)

    if task == "01":
        comm = [k for k in kernels if "nccl" in k["name"].lower()]
        compute = [
            k
            for k in kernels
            if (
                "gemm" in k["name"].lower()
                or "matmul" in k["name"].lower()
                or k["name"].startswith("nvjet_")
            )
        ]
        result = _summarize_pair(comm, compute, "comm_kernel", "compute_kernel")
    elif task == "04":
        compute = [k for k in kernels if "busy_compute" in k["name"]]
        result = _summarize_pair(memcpys, compute, "memcpy", "compute_kernel")
    elif task == "02":
        partial = [k for k in kernels if "partial_gemm_peer_write" in k["name"]]
        by_device: dict[int, list[dict[str, Any]]] = {}
        for item in partial:
            by_device.setdefault(int(item["device"]), []).append(item)
        devices = sorted(by_device)
        peer_overlap_us = 0.0
        if len(devices) >= 2:
            peer_overlap_us = _intersection_us(
                [(item["start_ns"], item["end_ns"]) for item in by_device[devices[0]]],
                [(item["start_ns"], item["end_ns"]) for item in by_device[devices[1]]],
            )
        result = {
            "kernel_count": len(kernels),
            "memcpy_count": len(memcpys),
            "partial_gemm_peer_write_count": len(partial),
            "partial_gemm_peer_write_cross_device_overlap_us": peer_overlap_us,
            "kernels": kernels,
            "memcpys": memcpys,
        }
    else:
        fused = [k for k in kernels if "comm_compute_kernel" in k["name"]]
        result = {
            "kernel_count": len(kernels),
            "memcpy_count": len(memcpys),
            "comm_compute_kernel_count": len(fused),
            "comm_compute_kernels": fused,
            "kernels": kernels,
            "memcpys": memcpys,
        }

    result["task"] = task
    result["sqlite"] = sqlite_path.name
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--task", choices=["01", "02", "03", "04"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(args.sqlite_path, args.task)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
