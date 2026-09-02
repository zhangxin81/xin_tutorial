#!/usr/bin/env python3
"""示例一：NCCL collective 与独立 GEMM 在不同 CUDA stream 上重叠。

对应文章《GPU 计算与通信融合入门》第六章。这是四种融合路线中最粗粒度的一种：
kernel / stream 级并发，通信 kernel 通常仍会占用少量 SM。

要点：
  - all_gather_into_tensor 提交到通信 stream，与其结果无关的 GEMM 提交到计算
    stream。`async_op=True` 只表示 CPU 侧异步提交，不保证 GPU timeline 物理
    重叠；是否真的重叠要用 Nsight Systems 查看。
  - PyTorch 要求：异步 collective 的输出被其他 stream 消费前，必须先
    work.wait()（参见 PyTorch distributed 文档的 stream 语义说明）。

运行（本目录自带独立环境配置，见 pyproject.toml / README.md）：
  pip install .
  torchrun --standalone --nproc-per-node=2 01_stream_overlap.py
  # 自定义规模：
  torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
      --rows-per-rank 4096 --hidden 4096

 profiling：
  nsys profile -o 01_stream_overlap --force-overwrite true \
      torchrun --standalone --nproc-per-node=2 01_stream_overlap.py
  nsys-ui 01_stream_overlap.nsys-rep   # 检查 NCCL kernel 与 GEMM 是否并发

硬件要求：至少 2 张 NVIDIA GPU（同一台机器），PyTorch 带 NCCL 支持。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NCCL all-gather 与独立 GEMM 的 stream 级重叠示例"
    )
    parser.add_argument(
        "--rows-per-rank", type=int, default=2048,
        help="每个 rank 本地的行数（默认 2048，减小可降低显存与耗时）",
    )
    parser.add_argument(
        "--hidden", type=int, default=2048,
        help="隐藏维大小（默认 2048）",
    )
    parser.add_argument(
        "--gemm-m",
        type=int,
        default=None,
        help="独立 GEMM 的 M 维；默认等于 --hidden",
    )
    parser.add_argument(
        "--gemm-n",
        type=int,
        default=None,
        help="独立 GEMM 的 N 维；默认等于 --hidden",
    )
    parser.add_argument(
        "--gemm-k",
        type=int,
        default=None,
        help="独立 GEMM 的 K 维；默认等于 --hidden",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="随机数种子，保证各 rank 一致性检查可复现",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=2,
        help="正式计时前的预热轮数（默认 2）",
    )
    parser.add_argument(
        "--iters", type=int, default=3,
        help="正式计时轮数；若开启 --profile，仅最后一轮导出 trace（默认 3）",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="开启 torch profiler，并为每个 rank 导出 chrome trace",
    )
    parser.add_argument(
        "--profile-dir", type=str, default="profiles",
        help="profile 输出目录（默认 ./profiles）",
    )
    parser.add_argument(
        "--overlap-demo",
        action="store_true",
        help="使用更容易在 kernel timeline 中看到重叠的推荐规模",
    )
    parser.add_argument(
        "--require-kernel-overlap-us",
        type=float,
        default=0.0,
        help="若开启 --profile，要求每个 rank 的 CUDA kernel overlap 至少达到该微秒数",
    )
    return parser.parse_args()


def _merged_duration_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0

    merged: list[list[float]] = []
    for start_us, end_us in sorted(intervals):
        if not merged or start_us > merged[-1][1]:
            merged.append([start_us, end_us])
        else:
            merged[-1][1] = max(merged[-1][1], end_us)
    return sum(end_us - start_us for start_us, end_us in merged)


def _intersection_duration_us(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    total = 0.0
    i = 0
    j = 0
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    while i < len(left_sorted) and j < len(right_sorted):
        left_start, left_end = left_sorted[i]
        right_start, right_end = right_sorted[j]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end < right_end:
            i += 1
        else:
            j += 1
    return total


def _kernel_record(event: dict[str, Any]) -> dict[str, Any]:
    start_us = float(event["ts"])
    duration_us = float(event.get("dur", 0.0))
    args = event.get("args", {})
    return {
        "name": event.get("name", ""),
        "stream": args.get("stream"),
        "device": args.get("device"),
        "start_us": start_us,
        "end_us": start_us + duration_us,
        "duration_us": duration_us,
    }


def summarize_kernel_overlap_from_trace(trace_path: Path) -> dict[str, Any]:
    trace_text = trace_path.read_text(encoding="utf-8")
    try:
        trace = json.loads(trace_text)
    except json.JSONDecodeError:
        # Some PyTorch profiler builds emit an empty distributed-process-group
        # field as `"Process Group Description": ,`, which is not valid JSON.
        trace = json.loads(
            trace_text.replace('"Process Group Description": ,', '"Process Group Description": null,')
        )

    comm_kernels: list[dict[str, Any]] = []
    compute_kernels: list[dict[str, Any]] = []
    for event in trace.get("traceEvents", []):
        category = str(event.get("cat", "")).lower()
        if "kernel" not in category or "ts" not in event:
            continue

        name = event.get("name", "")
        args = event.get("args", {})
        lower_name = name.lower()
        record = _kernel_record(event)
        if "nccl" in lower_name or args.get("Collective name"):
            comm_kernels.append(record)
        else:
            compute_kernels.append(record)

    comm_intervals = [(item["start_us"], item["end_us"]) for item in comm_kernels]
    compute_intervals = [(item["start_us"], item["end_us"]) for item in compute_kernels]
    overlap_us = _intersection_duration_us(comm_intervals, compute_intervals)
    all_intervals = comm_intervals + compute_intervals
    if all_intervals:
        active_span_us = max(end_us for _, end_us in all_intervals) - min(
            start_us for start_us, _ in all_intervals
        )
    else:
        active_span_us = 0.0
    comm_us = _merged_duration_us(comm_intervals)
    compute_us = _merged_duration_us(compute_intervals)

    return {
        "trace": trace_path.name,
        "comm_kernel_count": len(comm_kernels),
        "compute_kernel_count": len(compute_kernels),
        "comm_kernel_us": comm_us,
        "compute_kernel_us": compute_us,
        "kernel_overlap_us": overlap_us,
        "active_span_us": active_span_us,
        "kernel_overlap_pct_of_comm": 100.0 * overlap_us / comm_us if comm_us > 0 else 0.0,
        "kernel_overlap_pct_of_compute": (
            100.0 * overlap_us / compute_us if compute_us > 0 else 0.0
        ),
        "kernel_overlap_pct_of_active_span": (
            100.0 * overlap_us / active_span_us if active_span_us > 0 else 0.0
        ),
        "comm_kernels": comm_kernels,
        "compute_kernels": compute_kernels,
    }


def summarize_overlap_ms(
    anchor: torch.cuda.Event,
    comm_start: torch.cuda.Event,
    comm_end: torch.cuda.Event,
    compute_start: torch.cuda.Event,
    compute_end: torch.cuda.Event,
) -> dict[str, float]:
    comm_start_ms = anchor.elapsed_time(comm_start)
    comm_end_ms = anchor.elapsed_time(comm_end)
    compute_start_ms = anchor.elapsed_time(compute_start)
    compute_end_ms = anchor.elapsed_time(compute_end)
    overlap_start_ms = max(comm_start_ms, compute_start_ms)
    overlap_end_ms = min(comm_end_ms, compute_end_ms)
    overlap_ms = max(0.0, overlap_end_ms - overlap_start_ms)
    active_span_ms = max(comm_end_ms, compute_end_ms) - min(comm_start_ms, compute_start_ms)
    return {
        "comm_start_ms": comm_start_ms,
        "comm_end_ms": comm_end_ms,
        "comm_ms": comm_end_ms - comm_start_ms,
        "compute_start_ms": compute_start_ms,
        "compute_end_ms": compute_end_ms,
        "compute_ms": compute_end_ms - compute_start_ms,
        "overlap_ms": overlap_ms,
        "active_span_ms": active_span_ms,
        "overlap_pct": 100.0 * overlap_ms / active_span_ms if active_span_ms > 0 else 0.0,
    }


def run_overlap_step(
    *,
    rank: int,
    local_x: torch.Tensor,
    gathered: torch.Tensor,
    independent_x: torch.Tensor,
    weight: torch.Tensor,
    comm_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    default_stream = torch.cuda.current_stream(device)
    anchor = torch.cuda.Event(enable_timing=True)
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)

    anchor.record(default_stream)
    comm_stream.wait_event(anchor)
    compute_stream.wait_event(anchor)

    with torch.cuda.stream(comm_stream):
        torch.cuda.nvtx.range_push(f"rank{rank}:all_gather")
        comm_start.record()
        work = dist.all_gather_into_tensor(gathered, local_x, async_op=True)

    with torch.cuda.stream(compute_stream):
        torch.cuda.nvtx.range_push(f"rank{rank}:independent_gemm")
        compute_start.record()
        independent_y = independent_x @ weight
        compute_end.record()
        torch.cuda.nvtx.range_pop()

    # 等待 NCCL 完成。随后让默认流等待两条工作流。
    work.wait()
    with torch.cuda.stream(comm_stream):
        comm_end.record()
        torch.cuda.nvtx.range_pop()
    default_stream.wait_stream(comm_stream)
    default_stream.wait_stream(compute_stream)
    torch.cuda.synchronize(device)

    return independent_y, summarize_overlap_ms(
        anchor=anchor,
        comm_start=comm_start,
        comm_end=comm_end,
        compute_start=compute_start,
        compute_end=compute_end,
    )


def main() -> None:
    args = parse_args()
    if args.overlap_demo:
        args.rows_per_rank = 65536
        args.hidden = 4096
        args.gemm_m = 8192 if args.gemm_m is None else args.gemm_m
        args.gemm_n = 8192 if args.gemm_n is None else args.gemm_n
        args.gemm_k = 8192 if args.gemm_k is None else args.gemm_k

    if not torch.cuda.is_available():
        raise RuntimeError("需要 NVIDIA GPU")
    if not dist.is_nccl_available():
        raise RuntimeError("当前 PyTorch 构建不支持 NCCL 后端")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters 不能为负数")
    if args.iters <= 0:
        raise ValueError("--iters 必须为正数")

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 超出当前可见 GPU 数量 {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()

    rows_per_rank = args.rows_per_rank
    hidden = args.hidden
    gemm_m = args.gemm_m if args.gemm_m is not None else hidden
    gemm_n = args.gemm_n if args.gemm_n is not None else hidden
    gemm_k = args.gemm_k if args.gemm_k is not None else hidden
    if min(rows_per_rank, hidden, gemm_m, gemm_n, gemm_k) <= 0:
        raise ValueError("所有矩阵维度都必须为正数")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # local_x 的值编码了 rank：第 r 段 all-gather 后应全为 r+1，
    # 便于做逐元素精确校验（fp16 下 exact 相等，rtol=0/atol=0）。
    local_x = torch.full(
        (rows_per_rank, hidden), float(rank + 1),
        device=device, dtype=torch.float16,
    )
    gathered = torch.empty(
        (world * rows_per_rank, hidden), device=device, dtype=torch.float16
    )

    # 这组 GEMM 与 all-gather 的输出无关，因此可以合法重叠。
    independent_x = torch.randn((gemm_m, gemm_k), device=device, dtype=torch.float16)
    weight = torch.randn((gemm_k, gemm_n), device=device, dtype=torch.float16)

    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device)
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    if args.profile and rank == 0:
        profile_dir.mkdir(parents=True, exist_ok=True)
    if args.profile:
        dist.barrier(device_ids=[local_rank])

    timed_summaries: list[dict[str, float]] = []
    independent_y = None

    for _ in range(args.warmup_iters):
        dist.barrier(device_ids=[local_rank])
        independent_y, _ = run_overlap_step(
            rank=rank,
            local_x=local_x,
            gathered=gathered,
            independent_x=independent_x,
            weight=weight,
            comm_stream=comm_stream,
            compute_stream=compute_stream,
            device=device,
        )

    for it in range(args.iters):
        dist.barrier(device_ids=[local_rank])
        enable_profile = args.profile and it == args.iters - 1
        if enable_profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                independent_y, summary = run_overlap_step(
                    rank=rank,
                    local_x=local_x,
                    gathered=gathered,
                    independent_x=independent_x,
                    weight=weight,
                    comm_stream=comm_stream,
                    compute_stream=compute_stream,
                    device=device,
                )
            prof.export_chrome_trace(str(profile_dir / f"rank{rank}_trace.json"))
            (profile_dir / f"rank{rank}_overlap.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            kernel_summary = summarize_kernel_overlap_from_trace(
                profile_dir / f"rank{rank}_trace.json"
            )
            (profile_dir / f"rank{rank}_kernel_overlap.json").write_text(
                json.dumps(kernel_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            independent_y, summary = run_overlap_step(
                rank=rank,
                local_x=local_x,
                gathered=gathered,
                independent_x=independent_x,
                weight=weight,
                comm_stream=comm_stream,
                compute_stream=compute_stream,
                device=device,
            )
        timed_summaries.append(summary)

    # 正确性检查：第 r 段应全为 r+1。
    for r in range(world):
        shard = gathered[r * rows_per_rank:(r + 1) * rows_per_rank]
        expected = torch.full_like(shard, float(r + 1))
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)
    assert torch.isfinite(independent_y).all().item()

    rank_summary = {
        key: sum(item[key] for item in timed_summaries) / len(timed_summaries)
        for key in timed_summaries[0]
    }
    all_rank_summaries = [None for _ in range(world)]
    dist.all_gather_object(all_rank_summaries, rank_summary)
    all_rank_kernel_summaries = [None for _ in range(world)]
    if args.profile:
        local_kernel_summary = json.loads(
            (profile_dir / f"rank{rank}_kernel_overlap.json").read_text(
                encoding="utf-8"
            )
        )
        dist.all_gather_object(all_rank_kernel_summaries, local_kernel_summary)
        if args.require_kernel_overlap_us > 0:
            below_requirement = [
                (other_rank, summary["kernel_overlap_us"])
                for other_rank, summary in enumerate(all_rank_kernel_summaries)
                if summary["kernel_overlap_us"] < args.require_kernel_overlap_us
            ]
            if below_requirement:
                details = ", ".join(
                    f"rank {other_rank}: {overlap_us:.3f} us"
                    for other_rank, overlap_us in below_requirement
                )
                raise RuntimeError(
                    "CUDA kernel overlap below requirement "
                    f"{args.require_kernel_overlap_us:.3f} us ({details})"
                )

    if rank == 0:
        print(
            f"PASS on {world} GPUs "
            f"(rows_per_rank={rows_per_rank}, hidden={hidden}, "
            f"gemm=({gemm_m}, {gemm_k})x({gemm_k}, {gemm_n}), "
            f"warmup={args.warmup_iters}, iters={args.iters})"
        )
        for other_rank, summary in enumerate(all_rank_summaries):
            print(
                f"rank {other_rank}: "
                f"comm={summary['comm_ms']:.3f} ms "
                f"[{summary['comm_start_ms']:.3f}, {summary['comm_end_ms']:.3f}], "
                f"compute={summary['compute_ms']:.3f} ms "
                f"[{summary['compute_start_ms']:.3f}, {summary['compute_end_ms']:.3f}], "
                f"overlap={summary['overlap_ms']:.3f} ms "
                f"({summary['overlap_pct']:.1f}% of active span)"
            )
        print("overlap>0 说明通信窗口与独立 GEMM 窗口存在交叠。")
        if args.profile:
            for other_rank, summary in enumerate(all_rank_kernel_summaries):
                print(
                    f"rank {other_rank} kernel trace: "
                    f"comm_kernels={summary['comm_kernel_count']} "
                    f"compute_kernels={summary['compute_kernel_count']} "
                    f"kernel_overlap={summary['kernel_overlap_us'] / 1000.0:.3f} ms "
                    f"({summary['kernel_overlap_pct_of_active_span']:.1f}% of active span)"
                )
            print(f"profile traces written to: {profile_dir}")
            print("rank*_kernel_overlap.json dump 记录真实 CUDA kernel 级重叠。")

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
