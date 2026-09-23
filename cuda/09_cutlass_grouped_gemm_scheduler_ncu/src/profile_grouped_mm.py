#!/usr/bin/env python3
"""Steady-state CUDA-event timing for torch._grouped_mm on Hopper.

Reconstruction of the torch-side benchmark behind this task's published
timing numbers (the original in-worker script is not public). Method:
allocate one contiguous operand buffer per group, call torch._grouped_mm
in a loop between two CUDA events, report the mean over `repeat` runs.

Usage:
  python3 profile_grouped_mm.py --mode grouped_bf16 --groups 10 \
      --m 128 --k 1024 --n 2048 --warmup 80 --repeat 3000
"""

import argparse
import json
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="grouped_bf16", choices=["grouped_bf16"])
    p.add_argument("--groups", type=int, default=10)
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--repeat", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    torch.cuda.init()
    dev = torch.device("cuda")

    g, m, k, n = args.groups, args.m, args.k, args.n
    a = torch.randn(g * m, k, device=dev, dtype=torch.bfloat16) * 0.1
    b = torch.randn(g, k, n, device=dev, dtype=torch.bfloat16) * 0.1
    offs = torch.arange(m, g * m + 1, m, device=dev, dtype=torch.int32)

    for _ in range(args.warmup):
        torch._grouped_mm(a, b, offs=offs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        torch._grouped_mm(a, b, offs=offs)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / args.repeat
    tflops = 2.0 * g * m * n * k / (ms * 1e-3) / 1e12
    print(json.dumps({
        "mode": args.mode,
        "groups": g, "m": m, "k": k, "n": n,
        "warmup": args.warmup, "repeat": args.repeat,
        "ms": ms,
        "tflops": tflops,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }, indent=2))


if __name__ == "__main__":
    main()
