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
import os

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
        "--seed", type=int, default=0, help="随机数种子，保证各 rank 一致性检查可复现",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 NVIDIA GPU")
    if not dist.is_nccl_available():
        raise RuntimeError("当前 PyTorch 构建不支持 NCCL 后端")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rows_per_rank = args.rows_per_rank
    hidden = args.hidden
    torch.manual_seed(args.seed)

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
    independent_x = torch.randn((hidden, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden, hidden), device=device, dtype=torch.float16)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device)
    # 两条工作流都要等默认流上 tensor 的创建/初始化完成。
    comm_stream.wait_stream(default_stream)
    compute_stream.wait_stream(default_stream)

    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(comm_stream):
        comm_start.record()
        work = dist.all_gather_into_tensor(gathered, local_x, async_op=True)
        # work.wait() 放到消费 gathered 之前；这里先让 CPU 去提交独立计算。

    with torch.cuda.stream(compute_stream):
        compute_start.record()
        independent_y = independent_x @ weight
        compute_end.record()

    # 等待 NCCL 完成。随后让默认流等待两条工作流。
    work.wait()
    with torch.cuda.stream(comm_stream):
        comm_end.record()
    default_stream.wait_stream(comm_stream)
    default_stream.wait_stream(compute_stream)
    torch.cuda.synchronize(device)

    # 正确性检查：第 r 段应全为 r+1。
    for r in range(world):
        shard = gathered[r * rows_per_rank:(r + 1) * rows_per_rank]
        expected = torch.full_like(shard, float(r + 1))
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)
    assert torch.isfinite(independent_y).all().item()

    if rank == 0:
        comm_ms = comm_start.elapsed_time(comm_end)
        compute_ms = compute_start.elapsed_time(compute_end)
        print(f"PASS on {world} GPUs "
              f"(rows_per_rank={rows_per_rank}, hidden={hidden})")
        print(f"all-gather stream time: {comm_ms:.3f} ms")
        print(f"independent GEMM time: {compute_ms:.3f} ms")
        print("是否真正重叠，请用 Nsight Systems 查看两条 GPU timeline。")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
