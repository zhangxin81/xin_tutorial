#!/usr/bin/env python3
"""Run a small concurrency sweep and write one CSV summary.

MPS must be started separately before running multiprocess cases.
Edit BASE_ARGS for the target model and workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", choices=["encoder", "causal_lm_forward", "generate"], default="encoder")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--include-multiprocess", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = [("baseline", 1)]
    cases += [("multistream", n) for n in (2, 4, 8)]
    if args.include_multiprocess:
        cases += [("multiprocess", n) for n in (2, 4)]

    rows = []
    for mode, concurrency in cases:
        result_path = out / f"{mode}_c{concurrency}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("benchmark.py")),
            "--mode", mode,
            "--model", args.model,
            "--task", args.task,
            "--dtype", args.dtype,
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--requests", str(args.requests),
            "--warmup", str(args.warmup),
            "--concurrency", str(concurrency),
            "--arrival", "closed_loop",
            "--output", str(result_path),
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row = {k: v for k, v in payload.items() if k != "metadata"}
        row.update({f"meta_{k}": v for k, v in payload["metadata"].items()})
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
