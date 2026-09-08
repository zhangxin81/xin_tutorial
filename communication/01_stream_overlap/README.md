# 任务 01：NCCL 与 GEMM 流重叠（stream overlap）

教程《GPU 计算与通信融合入门》第六章配套代码。四种融合路线中最粗粒度的一种：
kernel / stream 级并发，通信 kernel 通常仍会占用少量 SM。

把 `all_gather_into_tensor` 放入通信 stream，把与其结果无关的 GEMM 放入计算
stream。`async_op=True` 只表示 CPU 侧异步提交，**不保证 GPU timeline 物理重叠**；
PyTorch 要求异步 collective 的输出被其他 stream 消费前必须先 `work.wait()`。

## 环境要求（独立于其他任务）

| 项 | 要求 |
|---|---|
| GPU | 同一节点 ≥2 张 NVIDIA GPU |
| 网络 | 无特殊要求（单机 NCCL） |
| Python | ≥3.9 |
| PyTorch | ≥2.1，需带 NCCL 支持（Linux 官方 wheel 默认包含） |
| 其他 | 无（纯 PyTorch，无需编译） |

## 安装（本目录独立环境）

```bash
cd 01_stream_overlap
python3 -m venv .venv && source .venv/bin/activate
pip install .          # 依赖见 pyproject.toml（torch>=2.1）
```

## 运行

```bash
torchrun --standalone --nproc-per-node=2 01_stream_overlap.py

# 自定义规模（减小可降低显存与耗时）
torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --rows-per-rank 4096 --hidden 4096

# 导出每个 rank 的 chrome trace，并打印计算/通信 overlap 窗口
torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --rows-per-rank 4096 --hidden 4096 \
    --warmup-iters 2 --iters 4 --profile --profile-dir ./profiles

# 推荐的重叠观测配置：通信量和 GEMM 都更大，更容易在 kernel timeline 看到重叠
torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --overlap-demo \
    --warmup-iters 2 --iters 4 \
    --profile --profile-dir ./profiles \
    --require-kernel-overlap-us 100
```

参数：
- `--rows-per-rank`（默认 2048）
- `--hidden`（默认 2048）
- `--gemm-m` / `--gemm-n` / `--gemm-k`：独立 GEMM 的矩阵维度，默认都等于
  `--hidden`
- `--seed`（默认 0）
- `--warmup-iters`（默认 2）：预热轮数
- `--iters`（默认 3）：正式计时轮数
- `--profile`：开启 `torch.profiler` 并导出 chrome trace
- `--profile-dir`（默认 `./profiles`）：trace 和 overlap JSON 输出目录
- `--overlap-demo`：覆盖为推荐规模，用于稳定观测 kernel 级 overlap
- `--require-kernel-overlap-us`：开启 `--profile` 时，要求每个 rank 的真实 CUDA
  kernel overlap 至少达到指定微秒数，否则程序报错

## 正确性校验

程序结束前做逐元素精确校验：all-gather 结果第 r 段应全为 r+1（fp16 exact），
独立 GEMM 输出必须有限。打印 `PASS` 才算通过。

## profiling

脚本内部 trace 检查：

```bash
torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --overlap-demo \
    --warmup-iters 2 --iters 4 \
    --profile --profile-dir ./profiles \
    --require-kernel-overlap-us 100
```

Nsight Systems report：

```bash
nsys profile -o report01 --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --overlap-demo \
    --warmup-iters 2 --iters 4
nsys-ui report01.nsys-rep
```

观察要点：
- 终端会打印每个 rank 的 `comm` / `compute` 时间窗口，以及 `overlap` 时长与占比。
- `./profiles/rank*.json` 是 `torch.profiler` 导出的 chrome trace，可用浏览器或 Perfetto 打开。
- `./profiles/rank*_kernel_overlap.json` 是脚本从 chrome trace 中解析出的真实 CUDA
  kernel overlap 摘要，包含 NCCL kernel、GEMM kernel 的时间区间和重叠时长。
- Nsight Systems 里可结合 NVTX 区间 `rank*:all_gather` 和 `rank*:independent_gemm`，观察 NCCL kernel 与 GEMM kernel 是否在两条 stream 的 timeline 上并发。
- 不要在外层 `nsys profile` 时同时传脚本的 `--profile`，否则 Nsight Systems 和
  `torch.profiler` 会同时订阅 CUPTI，内层 PyTorch trace 可能缺 CUDA activity。
