import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_producer(x, gamma, y, m: tl.constexpr, k: tl.constexpr,
                      eps: tl.constexpr, tail_iters: tl.constexpr,
                      BLOCK_K: tl.constexpr, USE_GDC: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < k
    xv = tl.load(x + row * k + offs, mask=mask, other=0.0)
    gv = tl.load(gamma + offs, mask=mask, other=0.0)
    ss = tl.sum(xv * xv, axis=0)
    inv_rms = tl.rsqrt(ss / k + eps)
    yv = xv * inv_rms * gv
    tl.store(y + row * k + offs, yv, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    tail = tl.full((), 0.0, tl.float32)
    for _ in tl.static_range(0, tail_iters):
        tail += tl.sum((xv * 1.000001 + gv * 0.000001) * 1.0e-7, axis=0)
    tl.store(y + row * k, tail * 0.0, mask=False)


@triton.jit
def _qkv_consumer(y, w, out, sink, m: tl.constexpr, k: tl.constexpr, n: tl.constexpr,
                  BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, USE_GDC: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    koffs = tl.arange(0, BLOCK_K)

    w_prefetch = tl.load(w + koffs[:, None] * n + cols[None, :],
                         mask=(koffs[:, None] < k) & (cols[None, :] < n),
                         other=0.0)
    pre = tl.sum(w_prefetch, axis=0)
    tl.store(sink + cols, pre * 1.0e-20, mask=cols < n)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    yv = tl.load(y + row * k + koffs, mask=koffs < k, other=0.0)
    wv = tl.load(w + koffs[:, None] * n + cols[None, :],
                 mask=(koffs[:, None] < k) & (cols[None, :] < n), other=0.0)
    acc = tl.sum(yv[:, None] * wv, axis=0)
    tl.store(out + row * n + cols, acc, mask=cols < n)


def supports_pdl() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    return hasattr(tl.extra.cuda, "gdc_wait") and hasattr(tl.extra.cuda, "gdc_launch_dependents")


def run_once(x, gamma, weight, m, k, n, tail_iters, use_pdl):
    y = torch.empty((m, k), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    sink = torch.empty((n,), device=x.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(k)
    block_n = 16
    _rmsnorm_producer[(m,)](x, gamma, y, m, k, 1.0e-6, tail_iters,
                            BLOCK_K=block_k, USE_GDC=use_pdl,
                            launch_pdl=use_pdl, num_warps=8)
    _qkv_consumer[(m, triton.cdiv(n, block_n))](y, weight, out, sink, m, k, n,
                                                 BLOCK_K=block_k, BLOCK_N=block_n,
                                                 USE_GDC=use_pdl,
                                                 launch_pdl=use_pdl, num_warps=8)
    return out


def bench(args, use_pdl):
    torch.manual_seed(0)
    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32) * 0.05
    gamma = torch.randn((args.k,), device="cuda", dtype=torch.float32) * 0.05 + 1.0
    weight = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32) * 0.05

    for _ in range(args.warmup):
        run_once(x, gamma, weight, args.m, args.k, args.n, args.tail_iters, use_pdl)
    torch.cuda.synchronize()

    samples = []
    out = None
    for _ in range(args.iters):
        start = time.perf_counter()
        out = run_once(x, gamma, weight, args.m, args.k, args.n, args.tail_iters, use_pdl)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "pdl", "both"], default="both")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--tail-iters", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cc = torch.cuda.get_device_capability()
    print(f"GPU={torch.cuda.get_device_name()} cc={cc[0]}.{cc[1]} triton={triton.__version__}")
    if args.mode in ("pdl", "both") and not supports_pdl():
        raise SystemExit("Triton PDL needs SM90+ and tl.extra.cuda.gdc_* support")

    baseline = pdl = None
    if args.mode in ("baseline", "both"):
        baseline_ms, baseline = bench(args, False)
        print(f"triton_baseline_ms={baseline_ms:.4f}")
    if args.mode in ("pdl", "both"):
        pdl_ms, pdl = bench(args, True)
        print(f"triton_pdl_ms={pdl_ms:.4f}")
    if args.mode == "both":
        max_abs = (baseline - pdl).abs().max().item()
        max_rel = ((baseline - pdl).abs() / baseline.abs().clamp_min(1.0e-6)).max().item()
        print(f"max_abs_error={max_abs:.8g} max_rel_error={max_rel:.8g}")


if __name__ == "__main__":
    main()
