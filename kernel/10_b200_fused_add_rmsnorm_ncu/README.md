# 任务 10:B200 上打完一条 fused add+RMSNorm 的完整优化路线(NCU 实战)

教程《24 小时冲上 NVIDIA kernel 榜单第 15》配套代码。算子是 Llama 类模型
decoder 层里的 `residual + hidden_states` 后接 RMSNorm(hidden size 固定
8192,BF16):一个没有矩阵乘、看起来"没什么可优化"的访存型算子。本任务
保留根目录的完整实现,加 4 个教程正文展开的实验(新编号 01–04)和
一套可复现的 benchmark/NCU 采集流程。

一句话:**每个版本的去留都有 NCU 报告或全量 benchmark 兜底,"看起来更快"
不算数。**

## 文件

| 文件 | 内容 |
|---|---|
| `reference.py` | 参考实现,correctness contract 的出处(两个 BF16 舍入点) |
| `kernel.cu` + `binding.cpp` + `submission.py` | 根实现:512/1024-thread shape dispatch + 按 rows 缓存的 CUDA Graph |
| `definition.json` / `workload.jsonl` | 算子定义与 16 个官方 workload |
| `build_and_run.sh` | 一键构建 + 三候选冒烟(official / 01 / 根实现) |
| `experiments/` | 4 个实验目录 + benchmark/NCU 脚本,见下节与 `experiments/README.md` |

## experiments/ 布局

| 目录/文件 | 一句话结论 |
|---|---|
| `01_triton_w16_maxnreg32/` | 寄存器 34→32,occupancy 64.92%→86.53%,无 spill,duration 42.08→40.35 us |
| `02_cute_dsl_copy128/` | 显式 128-bit copy atom,与 01 lowering 后形态几乎一致 |
| `03_cuda_graph_replay/` | CUDA Graph 按 rows 缓存,同轮 smoke 1.0188× vs 02;host 收益 NCU 测不到 |
| `04_cuda_tma_bulk_smem/` | TMA staging 负例,约 0.082 ms vs 直接路径 0.066 ms |
| `score_aligned_benchmark.py` | 全量 correctness + SOL 分数基准,16 workload 决策入口 |
| `profile_candidate.py` / `run_ncu.sh` | NCU 采集:稳定 shape 循环 + 稳态 capture |

## 核心证据速查

根实现(全部 B200、NCU 2025.2.1、`--set full` 采集):

| 观察 | rows=131 | rows=8192 |
|---|---:|---:|
| Duration | 7.776 us | 69.568 us |
| DRAM throughput | 0.555 TB/s | 5.329 TB/s |
| waves/SM | 0.22 | 13.84 |
| active warps | 21.93% | 88.09% |
| 主导 stall | — | long scoreboard |

SASS 证据(用 `run_ncu.sh` 重新生成):`LDG.E.128.CONSTANT`、
`HADD2.BF16_V2` / `HFMA2.BF16_V2`、`MUFU.RSQ`、`SHFL`、一次 `BAR.SYNC`、
`STG.E.128`——源码里写 `uint4` 只是源码事实,这组指令才是硬件事实。

01 与 02 的 SASS 同构:512 threads、32 registers、1.088 KB smem、
6×`LDG.E.128`、2×`STG.E.128`、2×`BAR.SYNC`。

## 环境要求(独立于其他任务)

| 项 | 要求 |
|---|---|
| GPU | 1 张 B200(SM100);A100/H100 可跑大部分版本,但寄存器/occupancy/SASS 证据会不同 |
| CUDA Toolkit | ≥12.8(`nvcc` 需支持 `compute_100`) |
| PyTorch | ≥2.x,CUDA 支持 BF16;triton ≥3.x(official/01) |
| CUTLASS Python DSL | 仅 `02_cute_dsl_copy128` 需要(python `cutlass` 包) |
| Nsight Compute | ≥2025.x CLI(`ncu`),性能计数器权限 |

## 构建与运行

```bash
cd kernel/10_b200_fused_add_rmsnorm_ncu
./build_and_run.sh                                          # 编译 + official/01/根实现 三候选冒烟
python3 experiments/score_aligned_benchmark.py --configs root_current   # 全量协议
```

首次运行会经 `torch.utils.cpp_extension.load` 即时编译 CUDA 扩展;16 个
workload 全部先过 correctness(逐元素与 `reference.py` 比对)再计时。

## Profile 命令

```bash
NCU_BIN=/path/to/ncu ./experiments/run_ncu.sh   # rows=8192 + 131,稳态 capture(--launch-skip 60)
rg 'LDG\.E\.128|HADD2|MUFU\.RSQ|SHFL|BAR\.SYNC|STG\.E\.128' experiments/ncu_out/*.sass.txt
```

`--set full` 会触发 Kernel Replay(本工程一次完整采集 39 passes),报告
里的 duration 是 profiler 环境数字,不能替代 benchmark;CUDA Graph 的
host 侧收益 NCU 单 kernel 也测不到,须回到 CUDA event / 全量基准。

## 边界说明

- 全部实测数字来自 B200(SM100)+ driver 580.126.20 + NCU 2025.2.1 的
  单次采集;换架构、驱动或工具版本,数值与 SASS 都可能变。
- 官方最佳已证明上传是 submission 49576,SOL Score 0.738275;根目录实现的
  本地估算(≈0.7353)属于稳定控制组,不等于更高的官方成绩。
