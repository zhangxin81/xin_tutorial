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
```

参数：`--rows-per-rank`（默认 2048）、`--hidden`（默认 2048）、`--seed`（默认 0）。

## 正确性校验

程序结束前做逐元素精确校验：all-gather 结果第 r 段应全为 r+1（fp16 exact），
独立 GEMM 输出必须有限。打印 `PASS` 才算通过。

## profiling

```bash
nsys profile -o report01 --force-overwrite true \
    torchrun --standalone --nproc-per-node=2 01_stream_overlap.py
nsys-ui report01.nsys-rep
```

观察要点：NCCL kernel 与 GEMM kernel 是否在两条 stream 的 timeline 上并发。
