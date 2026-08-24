# xin_tutorial — GPU 计算与通信融合入门（四套可运行代码）

Four runnable examples that go from coarse-grained stream overlap to near-zero-SM
hardware offload, companion code for the tutorial article
《GPU 计算与通信融合入门：术语、资源模型与四套可运行代码》.

跨 GPU 的“计算通信融合”至少分四层，粒度从粗到细：不同 stream 上的 kernel
overlap、GEMM prologue/epilogue 中嵌入 peer I/O、同一 kernel 内 warp/CTA
specialization、把数据面卸载到 copy engine / NIC DMA。本仓库为每一层提供一套
**可编译、可运行、自带正确性校验** 的教学代码：示例强调可理解、可检查、可
观测，不替代 NCCL、CUTLASS、DeepEP 或生产级 fused kernel。

## 示例总览

| 示例 | 文件 | 融合路线 | 并发粒度 | 是否占 SM | 依赖 |
|---|---|---|---|---|---|
| 一 | `examples/01_stream_overlap.py` | stream overlap | kernel / stream | 通信 kernel 通常占少量 SM | PyTorch + NCCL，≥2 GPU |
| 二 | `examples/02_fused_peer_write_rs.cu` | prologue/epilogue 融合 | CTA / tile | 计算 CTA 同时做 peer I/O | nvcc，≥2 GPU（P2P） |
| 三 | `examples/03_nvshmem_warp_specialization.cu` | kernel 内 warp 分工 | warp / pipeline stage | 通信和计算共享同一 CTA/SM | nvcc + NVSHMEM，≥2 GPU（P2P） |
| 四 | `examples/04_copy_engine_near_zero_sm.cu` | 硬件卸载 | copy engine / NIC | zero-SM 或 near-zero-SM | nvcc，≥2 GPU（P2P） |

四个示例分别演示：

1. **示例一** 把 `all_gather_into_tensor` 放入通信 stream、把与其结果无关的
   GEMM 放入计算 stream。`async_op=True` 只表示异步提交，不保证物理重叠；
   异步 collective 的输出被其他 stream 使用前必须显式 `work.wait()`。
2. **示例二** 两张 GPU 各算 K 维一半的 partial GEMM，epilogue 里每个 thread
   按 output row 判断 owner，直接把 partial result 写进 owner GPU 的 rank
   slot；两 rank 写不同 slot 因而不需要跨 GPU atomic，owner 最后做一次本地
   reduce——这就是 GEMM+ReduceScatter 的关键数据流。
3. **示例三** 单 kernel 启动 8 个 warp：warp 0 的 lane 0 对下一 PE 发起
   `nvshmem_float_put`，其余 7 个 warp 做独立 FMA 计算。这是 NVSHMEM 风格
   warp specialization 的最小教学版。
4. **示例四** `cudaMemcpyPeerAsync` 搬运 128 MiB P2P 数据，另一条 stream 跑
   FMA kernel；若驱动选择 copy engine 数据路径，payload 搬运不需要常驻通信
   CTA，即 near-zero-SM。程序会打印 `asyncEngineCount`，但该字段不构成
   overlap 保证。

四种方案可以组合：例如 AG+GEMM 可以用 copy engine 预取下一 tile，GEMM CTA
等 signal；MoE 可以让少量通信 CTA 做 dispatch，其余 SM 运行 grouped GEMM。

## 快速开始

硬件要求：同一台机器上至少 2 张支持 P2P 的 NVIDIA GPU（示例三还需安装
[NVSHMEM](https://developer.nvidia.com/nvshmem)）。

```bash
git clone https://github.com/zhangxin81/xin_tutorial.git
cd xin_tutorial

# 可选：安装配套工具（xin-tutorial check / info）
pip install -e .
xin-tutorial check        # 环境自检：GPU/P2P/NCCL/nvcc/NVSHMEM/nsys
xin-tutorial info         # 打印四个示例的运行命令
```

### 示例一：NCCL 与 GEMM 流重叠（PyTorch）

```bash
torchrun --standalone --nproc-per-node=2 examples/01_stream_overlap.py
# 自定义规模
torchrun --standalone --nproc-per-node=2 examples/01_stream_overlap.py \
    --rows-per-rank 4096 --hidden 4096
```

### 示例二：GEMM + ReduceScatter peer write（纯 CUDA）

```bash
nvcc -O2 -std=c++17 examples/02_fused_peer_write_rs.cu -o build/fused_peer_write_rs
./build/fused_peer_write_rs                 # 默认 M=N=K=512
./build/fused_peer_write_rs 1024 1024 1024  # 自定义规模（M、K 需为偶数）
```

### 示例三：NVSHMEM kernel 内 warp 分工

```bash
export NVSHMEM_HOME=/path/to/nvshmem
nvcc -O2 -std=c++17 -rdc=true \
    -I${NVSHMEM_HOME}/include examples/03_nvshmem_warp_specialization.cu \
    -L${NVSHMEM_HOME}/lib -lnvshmem_host -lnvshmem_device \
    -o build/nvshmem_warp_specialization
${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./build/nvshmem_warp_specialization
# 自定义规模：参数为 N（元素数）与计算迭代数
nvshmrun -np 2 ./build/nvshmem_warp_specialization 4194304 256
```

### 示例四：copy engine near-zero-SM

```bash
nvcc -O2 -std=c++17 examples/04_copy_engine_near_zero_sm.cu -o build/copy_engine_near_zero_sm
./build/copy_engine_near_zero_sm   # 默认 128 MiB；参数可改为其他 MiB 数
```

也可以直接用 Makefile：

```bash
make build    # 编译示例二/四（示例三用 make build03，需 NVSHMEM_HOME）
make run01    # torchrun 运行示例一
make run02 run04
make run03 NVSHMEM_HOME=/path/to/nvshmem   # 示例三：编译 + nvshmrun 运行
```

## 如何确认“真的融合了”

事件计时只能给出“可能重叠”的提示（并发耗时明显小于串行耗时）。要确认
物理重叠，需要看 GPU timeline：

```bash
# 示例一
nsys profile -o report01 --force-overwrite true \
    torchrun --standalone --nproc-per-node=2 examples/01_stream_overlap.py
# 示例二/四：nsys profile -o reportNN ./build/<binary>
# 示例三：nsys profile -o report03 ${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./build/...

nsys-ui report01.nsys-rep
```

观察要点：

- **Nsight Systems（timeline）**：NCCL kernel / memcpy 与 GEMM/FMA kernel 的
  行是否在时间轴上并发；示例四中 memcpy 是否落在 copy engine lane 而不是
  SM 上的 copy kernel。
- **Nsight Compute（单 kernel）**：示例二里远端写对 L2/PCIe/NVLink 流量的
  影响，warp stall 原因。

## 推荐学习顺序

1. 先运行示例一，学会区分 CPU 异步提交和 GPU timeline 真重叠。
2. 再运行示例四，观察 copy engine 与 SM kernel 的并发。
3. 用示例二理解 output owner、partial result 和 GEMM epilogue peer write。
4. 最后运行 NVSHMEM 示例，逐步加入 signal/wait 和双缓冲。
5. 准备进入生产实现时，再阅读 NCCL Device API、NVSHMEM fused GEMM 示例、
   [Flux](https://github.com/REDACTED/flux) 和
   [Triton-distributed](https://github.com/REDACTED/Triton-distributed)。

## 仓库结构

```
xin_tutorial/
├── examples/
│   ├── 01_stream_overlap.py               # 示例一：stream overlap（PyTorch）
│   ├── 02_fused_peer_write_rs.cu          # 示例二：GEMM+RS peer write
│   ├── 03_nvshmem_warp_specialization.cu  # 示例三：NVSHMEM warp 分工
│   └── 04_copy_engine_near_zero_sm.cu     # 示例四：copy engine near-zero-SM
├── xin_tutorial/                          # 配套 Python 包（环境自检 CLI）
│   ├── __init__.py
│   └── cli.py
├── Makefile
└── pyproject.toml
```

## License

MIT
