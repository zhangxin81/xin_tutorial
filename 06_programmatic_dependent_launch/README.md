# 任务 06:Programmatic Dependent Launch(PDL)——让下一个 kernel 提前启动

教程《在 H100 上看见 PDL》配套代码。PDL 是 Hopper(SM90+)引入的 CUDA 机制:
同一个 stream 里,后继 kernel 不必等前驱 kernel 整体结束,前驱在尾部显式发出
`cudaTriggerProgrammaticLaunchCompletion()` 后,后继就可以提前驻留并执行**不依赖
前驱输出**的 prologue,直到第一次读依赖数据前才在 `cudaGridDependencySynchronize()`
处等待。

一句话:**PDL 放宽的是启动时机,不放宽数据依赖。**

## 三个 demo

| 文件 | 演示内容 |
|---|---|
| `pdl_rmsnorm_qkv.cu` | 机制验证:`rmsnorm_producer` → `qkv_consumer<UsePdl>` 依赖链,`--tail-cycles` 把 producer 尾部人为拉长,让重叠窗口在 Nsight Systems 里可见 |
| `pdl_overlap_benefit.cu` | 收益构造:把 PDL 的收益条件拆成 `producer_tail_cycles` × `consumer_prologue_cycles` 两个旋钮,单边为零时收益归零,双边同时存在时收益最高约 47% |
| `pdl_rmsnorm_qkv_triton.py` | Triton 侧等价实现:`launch_pdl=True` + `tl.extra.cuda.gdc_launch_dependents()` / `gdc_wait()`,验证 Triton ≥3.5(含 PR #6394)对 PDL 的支持 |

两个 CUDA demo 的 kernel 都是朴素 FP32 实现,定位是"机制可读、可 profile",
不替代生产里的 cuBLASLt / CUTLASS / Quack kernel。

## 环境要求(独立于其他任务)

| 项 | 要求 |
|---|---|
| GPU | ≥1 张 compute capability 9.0+ 的 NVIDIA GPU(H100 等;程序启动时会检查) |
| CUDA Toolkit | ≥12.x(`nvcc`,实测 12.9) |
| 运行时依赖 | CUDA demo 无;triton demo 需 torch + triton(含 PR #6394,实测 3.5.1) |

## 构建与运行

```bash
cd 06_programmatic_dependent_launch
./build_and_run.sh                     # 编译并依次跑 rmsnorm sweep + benefit sweep + triton
./build_and_run.sh rmsnorm             # 只跑机制验证 sweep
./build_and_run.sh benefit             # 只跑收益构造 sweep
./build_and_run.sh triton              # 只跑 Triton 版
# 或手动:
nvcc -O3 -std=c++17 -arch=sm_90a pdl_rmsnorm_qkv.cu -o pdl_rmsnorm_qkv
./pdl_rmsnorm_qkv --mode both --m 64 --k 4096 --n 768 --warmup 10 --iters 50 --tail-cycles 300000
```

每个 sweep 组都输出 `baseline_ms` / `pdl_ms` / `max_abs_error` / `max_rel_error`,
`both` 模式下两种启动方式的输出逐元素对比,预期 `max_abs_error=0`。

## 正确性边界:trigger 放哪

本任务两个 CUDA demo 的 producer 都把顺序写成:

```cuda
/* 写出 consumer 依赖的数据 y */ → __threadfence() → cudaTriggerProgrammaticLaunchCompletion() → 不影响正确性的尾部
```

trigger 之后 producer 还会继续跑,但 consumer 的 `cudaGridDependencySynchronize()`
只保证 **trigger 之前** 的内存写入对它可见。所以 trigger 必须放在 `y` 写完、
且 `__threadfence()` 落地之后;consumer 对 `y` 的第一次读取必须在 wait 之后,
并且不能让编译器把它提到 wait 之前。

## H100 实测结果(2026-09-07,sm_90a,driver 535.129.03)

**机制验证**(m=64, k=4096, n=768, tail_cycles=300000,warmup 10 / iters 50,
Nsight Systems pair 级平均,跳过第一组):

| mode | producer_us | consumer_us | gap_us | overlap_us | pair_us |
|---|---:|---:|---:|---:|---:|
| baseline | 159.65 | 615.82 | +1.22 | 0 | 776.69 |
| PDL | 160.11 | 754.00 | −152.06 | 152.06 | 762.05 |

时间线上 consumer 提前 152 us 启动,但 pair 只快 14.6 us(1.9%):consumer
自身变长约 138 us(提前启动后在 wait 处等待 + 与 producer 尾部并发驻留的资源
竞争 + PDL 路径开销)。**可见重叠 ≠ 净收益。**

**收益构造**(blocks=1, threads=128,body=100000):

| producer_tail | consumer_prologue | baseline_ms | pdl_ms | speedup |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0603 | 0.0589 | 2.3% |
| 500000 | 0 | 0.3131 | 0.3103 | 0.9% |
| 0 | 1000000 | 0.5655 | 0.5640 | 0.3% |
| 500000 | 500000 | 0.5653 | 0.3117 | 44.9% |
| 1000000 | 1000000 | 1.0704 | 0.5643 | 47.3% |

只有 producer 尾部与 consumer 独立 prologue **同时存在**,重叠才有东西可装;
单边为零时收益接近噪声。selected 配置(1M/1M)的 timeline:overlap 504.96 us,
pair 1070.35 → 564.52 us,可见重叠几乎全部转成净收益。Nsight Compute 显示该
配置 SM/DRAM throughput ≈ 0.07%/0%,active warps 6.25%(1 block),GPU 大量
资源空闲,所以重叠不被资源竞争吃掉。

**Triton**:3.5.1 上 `launch_pdl=True` + `gdc_wait()` 正确性 PASS
(`max_abs_error=0`),计时与 CUDA demo 同量级;本机 worker 上 Triton 的
nsys profile 卡住,timeline 证据以 CUDA C++ 版为准。

## Profile 命令

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  -o baseline ./pdl_rmsnorm_qkv --mode baseline --m 64 --k 4096 --n 768 \
  --warmup 2 --iters 5 --tail-cycles 300000

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  -o pdl ./pdl_rmsnorm_qkv --mode pdl --m 64 --k 4096 --n 768 \
  --warmup 2 --iters 5 --tail-cycles 300000
```

在 GPU timeline 放大一对相邻 kernel:baseline 中 consumer 在 producer 结束后
启动;PDL 中 consumer 的起点落在 producer 结束之前。注意 NVIDIA 明确说明重叠
是机会性的,不保证每次发生;nsys 看不到重叠时要结合端到端计时和重复性判断。

## 适用判断

- 相邻、同 stream、依赖关系清楚的 kernel pair(如 RMSNorm → QKV GEMM、连续
  MoE GEMM 的 epilogue/prologue 交界)。
- producer 尾部低资源、consumer 有真正独立的 prologue 时,收益接近
  `min(producer_tail, consumer_prologue)`;consumer 一启动就占满 SM/带宽,或
  很快阻塞在 wait 上,重叠会被竞争和等待吃掉。
- 能 kernel fusion 时先比较 fusion(还能省一次全局内存往返);PDL 保留两个
  kernel,只压缩交界空档,不能替代 NCCL/多 stream 的通信计算重叠。
