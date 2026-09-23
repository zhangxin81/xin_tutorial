import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.algorithm import copy
from cutlass.cute.math import rsqrt
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.utils.smem_allocator import SmemAllocator


HIDDEN_SIZE = 8192
THREADS = 512
ELEMS_PER_THREAD = HIDDEN_SIZE // THREADS
PAIR = 8
GROUPS_PER_THREAD = ELEMS_PER_THREAD // PAIR
_EXECUTOR_CACHE = {}


@cute.kernel
def _fused_add_rmsnorm_cute(hidden, residual, weight, output, eps: cutlass.Float32):
    tx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    smem = SmemAllocator()
    warp_sums = smem.allocate_tensor(cutlass.Float32, cute.make_layout(16))
    xs = cute.make_rmem_tensor((ELEMS_PER_THREAD,), cutlass.Float32)
    hidden_vals = cute.make_rmem_tensor((PAIR,), cutlass.BFloat16)
    residual_vals = cute.make_rmem_tensor((PAIR,), cutlass.BFloat16)

    base = row * HIDDEN_SIZE
    thread_sum = cutlass.Float32(0.0)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )

    for group in cutlass.range_constexpr(GROUPS_PER_THREAD):
        group_base = base + group * THREADS * PAIR + tx * PAIR
        hidden_slice = cute.make_tensor((hidden.iterator + group_base).align(16), cute.make_layout(PAIR))
        residual_slice = cute.make_tensor((residual.iterator + group_base).align(16), cute.make_layout(PAIR))
        copy(copy_atom, hidden_slice, hidden_vals)
        copy(copy_atom, residual_slice, residual_vals)
        for j in cutlass.range_constexpr(PAIR):
            i = group * PAIR + j
            x = (hidden_vals[j].to(cutlass.Float32) + residual_vals[j].to(cutlass.Float32)).to(cutlass.BFloat16).to(cutlass.Float32)
            xs[i] = x
            thread_sum += x * x

    warp_sum = cute.arch.warp_reduction_sum(thread_sum)
    if lane == 0:
        warp_sums[warp] = warp_sum
    cute.arch.sync_threads()

    total = cutlass.Float32(0.0)
    if tx < 16:
        total = warp_sums[tx]
    total = cute.arch.warp_reduction_sum(total, threads_in_group=16)
    if tx == 0:
        warp_sums[0] = total
    cute.arch.sync_threads()

    inv_rms = rsqrt(warp_sums[0] * cutlass.Float32(1.0 / HIDDEN_SIZE) + eps, fastmath=True)

    weight_vals = cute.make_rmem_tensor((PAIR,), cutlass.BFloat16)
    output_vals = cute.make_rmem_tensor((PAIR,), cutlass.BFloat16)

    for group in cutlass.range_constexpr(GROUPS_PER_THREAD):
        col_base = group * THREADS * PAIR + tx * PAIR
        out_base = base + col_base
        weight_slice = cute.make_tensor((weight.iterator + col_base).align(16), cute.make_layout(PAIR))
        output_slice = cute.make_tensor((output.iterator + out_base).align(16), cute.make_layout(PAIR))
        copy(copy_atom, weight_slice, weight_vals)
        for j in cutlass.range_constexpr(PAIR):
            i = group * PAIR + j
            x = xs[i]
            normalized = (x * inv_rms).to(cutlass.BFloat16).to(cutlass.Float32)
            output_vals[j] = (normalized * weight_vals[j].to(cutlass.Float32)).to(cutlass.BFloat16)
        copy(copy_atom, output_vals, output_slice)


@cute.jit
def _launch(hidden, residual, weight, output, eps: cutlass.Float32, rows: cutlass.Int32):
    _fused_add_rmsnorm_cute(hidden, residual, weight, output, eps).launch(
        grid=[rows, 1, 1],
        block=[THREADS, 1, 1],
        smem=64,
    )


def _get_executor(numel, rows, eps):
    key = (int(numel), int(rows), float(eps))
    executor = _EXECUTOR_CACHE.get(key)
    if executor is not None:
        return executor

    hidden = make_fake_compact_tensor(cutlass.BFloat16, (int(numel),), assumed_align=16)
    residual = make_fake_compact_tensor(cutlass.BFloat16, (int(numel),), assumed_align=16)
    weight = make_fake_compact_tensor(cutlass.BFloat16, (HIDDEN_SIZE,), assumed_align=16)
    output = make_fake_compact_tensor(cutlass.BFloat16, (int(numel),), assumed_align=16)
    compiled = cute.compile(
        _launch,
        hidden,
        residual,
        weight,
        output,
        cutlass.Float32(float(eps)),
        cutlass.Int32(int(rows)),
    )
    executor = compiled.to(0)
    _EXECUTOR_CACHE[key] = executor
    return executor


@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // HIDDEN_SIZE
    flat_hidden = hidden_states.reshape(-1)
    flat_residual = residual.reshape(-1)
    flat_output = output.reshape(-1)
    executor = _get_executor(flat_hidden.numel(), rows, eps)
    executor(
        from_dlpack(flat_hidden),
        from_dlpack(flat_residual),
        from_dlpack(weight),
        from_dlpack(flat_output),
        cutlass.Float32(float(eps)),
        cutlass.Int32(rows),
    )
    return output
