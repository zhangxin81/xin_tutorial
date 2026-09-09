# xin_tutorial — 教程配套代码

本仓库收录我写的教程/科普文章中**附带的示例代码**，每篇文章的代码独立成
一个（或多个）任务目录，**持续不定时更新**——随新文章发布、旧示例修订而
增补，不设固定发布节奏。

## 门类

任务按主题归入**门类**（category）目录；任务编号全仓库连续、按发布顺序
递增——**目录管主题，编号管顺序**，所以旧文里提到的「任务 06」永远是同
一个目录，不会因归类而变号。

| 门类 | 收录范围 | 任务 |
|---|---|---|
| [`communication/`](communication/) | **通信**：多卡互联与计算通信融合——NCCL/NVSHMEM/P2P、集合通信、copy engine | 01~04 |
| [`cuda/`](cuda/) | **CUDA**：编程模型与运行时特性——stream/event、Graph、launch 机制、内存 API、NCU 剖析 | 05~08 |
| `kernel/`（规划中） | **Kernel**：kernel 编写与优化——warp primitives、Triton/CUTLASS、融合策略 | — |
| `fundamentals/`（规划中） | **基础**：体系结构与系统底座——SM/warp 结构、内存层级、带宽与延迟、数值格式 | — |
| `parallelism/`（规划中） | **并行策略**：模型与张量怎么切——DP/TP/SP/PP/EP、ZeRO/FSDP、分片与重分片 | — |
| `systems/`（规划中） | **系统**：推理/训练框架机制——continuous batching、PagedAttention、KV cache 管理与调度 | — |

新任务按文章主线就近归入现有门类；规划中的门类随各自**首个任务**落地建
目录；再往后仍有装不下的新主题（如性能分析方法论、数值算法）时照此扩展，
并同步更新本表与下方索引。编号从 09 继续往下排。

## 任务索引

### 通信 · communication/

| 目录 | 主题 | 来源教程 | 语言 | 额外依赖 |
|---|---|---|---|---|
| [`communication/01_stream_overlap/`](communication/01_stream_overlap/) | NCCL all-gather 与独立 GEMM 的 stream 级重叠 | 《GPU 计算与通信融合入门》第六章 | Python | PyTorch ≥2.1（带 NCCL） |
| [`communication/02_fused_peer_write_rs/`](communication/02_fused_peer_write_rs/) | GEMM+ReduceScatter：epilogue peer write 到 owner slot | 同上·第七章 | CUDA C++ | 无（仅 nvcc） |
| [`communication/03_nvshmem_warp_specialization/`](communication/03_nvshmem_warp_specialization/) | NVSHMEM 单 kernel 内通信/计算 warp 分工 | 同上·第八章 | CUDA C++ | NVSHMEM ≥2.x |
| [`communication/04_copy_engine_near_zero_sm/`](communication/04_copy_engine_near_zero_sm/) | copy engine near-zero-SM：peer DMA copy 与 SM 计算并发 | 同上·第九章 | CUDA C++ | 无（仅 nvcc） |

### CUDA · cuda/

| 目录 | 主题 | 来源教程 | 语言 | 额外依赖 |
|---|---|---|---|---|
| [`cuda/05_cuda_graph_pitfalls/`](cuda/05_cuda_graph_pitfalls/) | CUDA Graph capture/replay：三条硬约束的易错场景与修复 | 《CUDA Graph：一次录制、多次重放》 | CUDA C++ | 无（仅 nvcc，单 GPU） |
| [`cuda/06_programmatic_dependent_launch/`](cuda/06_programmatic_dependent_launch/) | PDL：同 stream 后继 kernel 提前启动与 producer/consumer 重叠 | 《在 H100 上看见 PDL》 | CUDA C++ + Python | triton demo 需 torch+triton ≥3.5 |
| [`cuda/07_gpu_concurrency_lab/`](cuda/07_gpu_concurrency_lab/) | 单卡并发组织：单/多 Stream 与多进程+MPS 在固定 P99 SLA 下的吞吐权衡 | —（独立实验，暂无配套教程） | Python + CUDA C++ | torch+transformers；MPS 需 Linux；单 GPU |
| [`cuda/08_h100_gemm_tile_hierarchy_ncu/`](cuda/08_h100_gemm_tile_hierarchy_ncu/) | 用 NCU+SASS 读出 cuBLASLt GEMM 的 threadblock/warp/thread 三级分块 | 《矩阵乘法在 H100 上是怎么分块计算的》 | CUDA C++ | 仅 nvcc+NCU；需 H100（SM90） |

两条内容主线，恰好对应现有两个门类：

- **通信**：01~04 合起来覆盖跨 GPU 计算通信融合的四种粒度（从粗到细）：
  stream overlap → GEMM prologue/epilogue 融合 → kernel 内 warp 分工 →
  copy engine 硬件卸载；
- **CUDA**：05、06 聚焦 kernel 交界处的开销（05 管 host 侧提交开销，06 管
  device 侧 kernel 间空档），07 再往上一层比单卡并发组织（多 Stream / 多进程
  +MPS）在固定 P99 SLA 下的吞吐上限；08 则往下钻进 kernel 内部，用 NCU 指令
  证据读出库 GEMM 的三级分块设计。全部单 GPU 即可运行。

## 使用方式

**每个任务目录完全独立**：有自己的 README（环境要求、配置、运行方法、参数、
正确性校验、profiling 命令）和自己的环境配置（Python 任务带 `pyproject.toml`，
CUDA 任务带 `build_and_run.sh` 一键构建脚本）。任务之间互不依赖，进入对应
目录按其 README 操作即可，例如：

```bash
cd communication/01_stream_overlap && cat README.md    # 环境要求与运行方法
cd communication/02_fused_peer_write_rs && ./build_and_run.sh
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
- 2026-09-02：新增任务 05_cuda_graph_pitfalls（配套《CUDA Graph：一次录制、
  多次重放》）：capture/replay 的三条硬约束各构造一个可复现 bug 与对应修复，
  附启动开销对比；单 GPU 即可运行。
- 2026-09-07：新增任务 06_programmatic_dependent_launch（配套《在 H100 上
  看见 PDL》）：CUDA C++ 与 Triton 双版本 producer/consumer 重叠 demo。
- 2026-09-08：新增任务 07_gpu_concurrency_lab（`cuda/`）：单 Stream / 多
  Stream / 多进程+MPS 三种并发组织在固定 P99 SLA 下的单卡吞吐对比基准，含
  block slot 占用微基准与多 Stream 输出一致性检查；单 GPU 可运行。
- 2026-09-08：建立门类目录结构：01~04 移入 `communication/`，05、06 移入
  `cuda/`，任务编号不变、git 历史保留；同时规划 `kernel/`、`fundamentals/`、
  `parallelism/`、`systems/` 四个门类，各自随首个任务落地。
- 2026-09-09：新增任务 08_h100_gemm_tile_hierarchy_ncu（`cuda/`，配套
  《矩阵乘法在 H100 上是怎么分块计算的》）：cuBLASLt 8192³ BF16 GEMM 的
  NCU 剖析，从 kernel 名、Launch Statistics、SASS 指令与 TMA 流量四路证据
  交叉验证 threadblock/warp/thread 三级 tile，附已验证采集的证据文件。

## License

MIT
