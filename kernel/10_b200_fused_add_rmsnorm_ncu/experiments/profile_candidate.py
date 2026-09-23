"""Run one candidate/shape in a tight loop for Nsight Compute.

The score benchmark is the source of truth for timing. This helper keeps the
same candidate loader and call signature, but gives NCU a simple repeated
region with stable shapes and already-built CUDA graph caches.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "experiments" / "score_aligned_benchmark.py"
EPS = 1e-5
HIDDEN = 8192


def _load_score_module():
    spec = importlib.util.spec_from_file_location("kernel69_score_for_profile", SCORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_shape(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be batch,seq")
    batch, seq = int(parts[0]), int(parts[1])
    if batch <= 0 or seq <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return batch, seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=260)
    parser.add_argument("--flush-l2", action="store_true")
    args = parser.parse_args()

    score = _load_score_module()
    if args.config not in score.DEFAULT_CANDIDATES:
        raise ValueError(f"unknown config: {args.config}")
    module = score.load_module(f"profile_{args.config}", score.DEFAULT_CANDIDATES[args.config])
    arity = len(inspect.signature(module.run).parameters)

    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (10, 0)
    torch.manual_seed(args.seed)
    batch, seq = args.shape
    hidden = torch.randn((batch, seq, HIDDEN), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(hidden)
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(hidden)
    cache = None
    if args.flush_l2:
        cache = torch.empty(torch.cuda.get_device_properties(0).L2_cache_size * 2, dtype=torch.int8, device="cuda")

    def launch():
        if cache is not None:
            cache.zero_()
        if arity >= 5:
            module.run(hidden, residual, weight, EPS, output)
        else:
            module.run(hidden, residual, weight, EPS)

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"{args.config}_{batch}x{seq}")
    for _ in range(args.iters):
        launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"profiled config={args.config} shape={batch}x{seq} iters={args.iters} flush_l2={args.flush_l2}")


if __name__ == "__main__":
    main()
