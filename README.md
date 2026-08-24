# xin_tutorial — 教程配套代码

本仓库收录我写的教程/科普文章中**附带的示例代码**，每篇文章的代码独立成
一个（或多个）任务目录，**持续不定时更新**——随新文章发布、旧示例修订而
增补，不设固定发布节奏。

## 任务索引

| 目录 | 主题 | 来源教程 | 语言 | 额外依赖 |
|---|---|---|---|---|
| [`01_stream_overlap/`](01_stream_overlap/) | NCCL all-gather 与独立 GEMM 的 stream 级重叠 | 《GPU 计算与通信融合入门》第六章 | Python | PyTorch ≥2.1（带 NCCL） |
| [`02_fused_peer_write_rs/`](02_fused_peer_write_rs/) | GEMM+ReduceScatter：epilogue peer write 到 owner slot | 同上·第七章 | CUDA C++ | 无（仅 nvcc） |
| [`03_nvshmem_warp_specialization/`](03_nvshmem_warp_specialization/) | NVSHMEM 单 kernel 内通信/计算 warp 分工 | 同上·第八章 | CUDA C++ | NVSHMEM ≥2.x |
| [`04_copy_engine_near_zero_sm/`](04_copy_engine_near_zero_sm/) | copy engine near-zero-SM：peer DMA copy 与 SM 计算并发 | 同上·第九章 | CUDA C++ | 无（仅 nvcc） |

01~04 合起来覆盖跨 GPU 计算通信融合的四种粒度（从粗到细）：stream overlap →
GEMM prologue/epilogue 融合 → kernel 内 warp 分工 → copy engine 硬件卸载。

## 使用方式

**每个任务目录完全独立**：有自己的 README（环境要求、配置、运行方法、参数、
正确性校验、profiling 命令）和自己的环境配置（Python 任务带 `pyproject.toml`，
CUDA 任务带 `build_and_run.sh` 一键构建脚本）。任务之间互不依赖，进入对应
目录按其 README 操作即可，例如：

```bash
cd 01_stream_overlap && cat README.md    # 环境要求与运行方法
cd 02_fused_peer_write_rs && ./build_and_run.sh
```

通用硬件底线：除特别说明外，任务需要同一节点 ≥2 张支持 P2P 的 NVIDIA GPU；
是否真的发生计算通信重叠，建议用 Nsight Systems 看 GPU timeline（各任务
README 里有对应的 nsys 命令）。

所有示例强调**可理解、可检查、可观测**（自带正确性校验与计时输出），定位是
教学，不替代 NCCL、CUTLASS、DeepEP 或生产级 fused kernel。

## 更新记录

不定期更新；新任务目录上线或已有任务修订都会记在这里。

- 2026-08-24：初始发布。任务 01~04（配套《GPU 计算与通信融合入门：术语、
  资源模型与四套可运行代码》），每个任务独立目录、独立环境说明。

## License

MIT
