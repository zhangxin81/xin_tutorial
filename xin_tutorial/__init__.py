"""xin_tutorial：GPU 计算与通信融合入门的配套工具包。

配套文章《GPU 计算与通信融合入门：术语、资源模型与四套可运行代码》，
提供运行环境自检（xin-tutorial check）与示例索引（xin-tutorial info）。
"""

__version__ = "0.1.0"

EXAMPLES = {
    "1": {
        "file": "examples/01_stream_overlap.py",
        "title": "NCCL collective 与独立 GEMM 的 stream 级重叠",
        "route": "stream overlap",
        "needs": [">=2 GPU", "PyTorch + NCCL"],
        "run": "torchrun --standalone --nproc-per-node=2 examples/01_stream_overlap.py",
    },
    "2": {
        "file": "examples/02_fused_peer_write_rs.cu",
        "title": "GEMM+ReduceScatter：epilogue 直接 peer write 到 owner slot",
        "route": "prologue/epilogue 融合",
        "needs": [">=2 GPU（P2P）", "nvcc"],
        "run": (
            "nvcc -O2 -std=c++17 examples/02_fused_peer_write_rs.cu -o build/fused_peer_write_rs\n"
            "          ./build/fused_peer_write_rs"
        ),
    },
    "3": {
        "file": "examples/03_nvshmem_warp_specialization.cu",
        "title": "NVSHMEM 单 kernel 内通信/计算 warp 分工",
        "route": "kernel 内 warp 分工",
        "needs": [">=2 GPU（P2P）", "nvcc + NVSHMEM"],
        "run": (
            "export NVSHMEM_HOME=/path/to/nvshmem\n"
            "          nvcc -O2 -std=c++17 -rdc=true -I${NVSHMEM_HOME}/include \\\n"
            "              examples/03_nvshmem_warp_specialization.cu \\\n"
            "              -L${NVSHMEM_HOME}/lib -lnvshmem_host -lnvshmem_device \\\n"
            "              -o build/nvshmem_warp_specialization\n"
            "          ${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./build/nvshmem_warp_specialization"
        ),
    },
    "4": {
        "file": "examples/04_copy_engine_near_zero_sm.cu",
        "title": "copy engine near-zero-SM：peer DMA copy 与 SM 计算并发",
        "route": "硬件卸载",
        "needs": [">=2 GPU（P2P）", "nvcc"],
        "run": (
            "nvcc -O2 -std=c++17 examples/04_copy_engine_near_zero_sm.cu -o build/copy_engine_near_zero_sm\n"
            "          ./build/copy_engine_near_zero_sm"
        ),
    },
}
