#!/usr/bin/env python3
"""xin-tutorial 命令行入口。

子命令：
  check  检查本机能否运行四个示例（GPU、PyTorch/NCCL、nvcc、NVSHMEM、P2P）。
  info   列出四个示例的文件、融合路线与运行命令。
"""

import argparse
import shutil
import sys


def _check_item(ok, label, detail=""):
    mark = "OK  " if ok else "MISS"
    line = f"[{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def cmd_check(_args: argparse.Namespace) -> int:
    """逐项检查运行示例一~四所需的环境，返回 0 表示全部就绪。"""
    print("== xin_tutorial 环境自检 ==")
    ready = True

    # 1) NVIDIA GPU 数量：直接依赖 torch；torch 缺失时只降级提示。
    gpu_count = None
    nccl_ok = False
    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _check_item(gpu_count > 0, "CUDA GPU (PyTorch 视角)",
                    f"可见 {gpu_count} 张" if gpu_count is not None else "")
        if gpu_count and gpu_count >= 2:
            names = []
            for i in range(min(gpu_count, 2)):
                p = torch.cuda.get_device_properties(i)
                names.append(f"GPU{i}={p.name}")
            can = torch.cuda.can_device_access_peer(0, 1)
            _check_item(can, "GPU0->GPU1 P2P", ", ".join(names))
        else:
            _check_item(False, "GPU0->GPU1 P2P", "需要 >=2 张 GPU 才能检查")
        ready = ready and gpu_count >= 2

        nccl_ok = torch.distributed.is_nccl_available()
        _check_item(nccl_ok, "PyTorch NCCL 后端",
                    torch.__version__ if nccl_ok else "示例一需要")
        ready = ready and nccl_ok
    except ImportError:
        _check_item(False, "PyTorch", "未安装；示例一和本检查需要（pip install torch）")
        ready = False

    # 2) 编译工具链：示例二/三/四需要 nvcc。
    nvcc = shutil.which("nvcc")
    _check_item(nvcc is not None, "nvcc", nvcc or "示例二/三/四需要 CUDA toolkit")
    ready = ready and nvcc is not None

    # 3) NVSHMEM：仅示例三需要。
    import os

    nvshmem_home = os.environ.get("NVSHMEM_HOME", "")
    has_nvshmem = bool(nvshmem_home) and os.path.isdir(
        os.path.join(nvshmem_home, "include")
    )
    _check_item(has_nvshmem, "NVSHMEM (NVSHMEM_HOME)",
                nvshmem_home or "仅示例三需要；从 NVIDIA 官网下载")

    # 4) profiler：强烈建议，用于确认是否真的物理重叠。
    nsys = shutil.which("nsys")
    _check_item(nsys is not None, "Nsight Systems (nsys)",
                nsys or "建议安装，用于观察 GPU timeline 是否重叠")

    print()
    if ready:
        print("环境就绪：示例一/二/四可直接运行"
              + ("，示例三需要 NVSHMEM。" if not has_nvshmem else "。"))
    else:
        print("环境未完全就绪：缺失项会影响对应示例，详见上方列表。")
    return 0


def cmd_info(_args: argparse.Namespace) -> int:
    """打印四个示例的索引卡片。"""
    from . import EXAMPLES

    print("GPU 计算与通信融合入门 — 四个示例（粒度从粗到细）：\n")
    for key, ex in EXAMPLES.items():
        print(f"示例{key}  [{ex['route']}]  {ex['title']}")
        print(f"  文件: {ex['file']}")
        print(f"  依赖: {'; '.join(ex['needs'])}")
        print("  运行:")
        for ln in ex["run"].splitlines():
            print(f"    {ln.strip()}")
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="xin-tutorial",
        description="GPU 计算与通信融合入门教程的配套工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="检查运行示例所需的环境")
    sub.add_parser("info", help="列出四个示例与运行命令")
    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check(args)
    return cmd_info(args)


if __name__ == "__main__":
    sys.exit(main())
