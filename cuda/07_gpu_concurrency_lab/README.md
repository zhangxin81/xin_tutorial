# 任务 07:单卡并发组织——单 Stream、多 Stream、多进程 + MPS

独立实验代码(暂无配套教程,成文后补链接)。要回答的问题很具体:**P99 SLA
固定不变,一张 GPU 换哪种并发组织、开到多少并发,单卡吞吐最高?** 三种模式
共用同一个基准脚本,只换"请求怎么进 GPU"这一层:

- `baseline`:一个 worker、默认 stream,请求串行排队——对照组;
- `multistream`:一个进程、一个模型实例,N 个 worker 各占一条
  `torch.cuda.Stream`,靠 kernel 级并发填 SM;
- `multiprocess`:N 个进程各自加载模型,MPS 把它们的 kernel 合到同一块 GPU
  上并发执行——进程级隔离,服务端统一调度。

一句话:**并发组织改的是排队结构,不改 GPU 的物理容量;胜出判据是 P99 不越线
之下的吞吐,不是 GPU utilization。** 配套 `block_slot_demo.cu` 微基准解释
"为什么并发不一定换来自适应的吞吐":驻留 block 数由 block slot / warp slot /
register / shared memory 四种上限共同决定,哪种先卡住,多出来的并发就在排队。

## 文件

| 文件 | 作用 |
|---|---|
| `benchmark.py` | 统一基准:三种模式 × 到达模式(closed_loop/burst/poisson),输出 JSON(吞吐 + P50/P95/P99/max 延迟 + 环境元数据) |
| `run_sweep.py` | 并发度 sweep(baseline、multistream 2/4/8、可选 multiprocess 2/4),汇总一张 CSV |
| `mps_control.sh` | MPS 启动/停止/状态查询;用私有 pipe 目录,避免多用户冲突 |
| `block_slot_demo.cu` | 微基准:两条 stream 各跑一个 busy kernel,扫 threads/smem 观察 CTA slot、warp slot、shared memory 谁先限制驻留 |
| `pyproject.toml` | Python 依赖(torch/transformers/accelerate/safetensors) |
| `build_and_run.sh` | 一键编译 block slot 微基准并跑三组代表配置 |

## 环境要求(独立于其他任务)

| 项 | 要求 |
|---|---|
| GPU | ≥1 张 NVIDIA GPU(单 GPU 任务,不需要 P2P/多卡) |
| Python | ≥3.9 |
| PyTorch | ≥2.3(带 CUDA 的官方 wheel) |
| transformers | ≥4.45 |
| CUDA Toolkit | block slot demo 需 `nvcc`;multiprocess 模式需 Linux 上的 `nvidia-cuda-mps-control` |
| 其他 | benchmark 主体纯 Python,无需编译 |

## 安装(本目录独立环境)

```bash
cd cuda/07_gpu_concurrency_lab
python3 -m venv .venv && source .venv/bin/activate
pip install .
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

建议固定并记录:GPU 型号、驱动、CUDA、PyTorch、Transformers、模型 commit、
dtype、batch、sequence length、输入分布、时钟锁定策略和功耗上限——并发对比
实验对环境漂移很敏感。

## 运行

单 Stream baseline(对照组):

```bash
python benchmark.py \
  --mode baseline \
  --model /path/to/model \
  --task encoder \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 128 \
  --requests 500 \
  --warmup 50 \
  --arrival closed_loop \
  --output results/baseline.json
```

Multi-Stream(一个模型实例 + N 条 stream):

```bash
python benchmark.py \
  --mode multistream \
  --model /path/to/model \
  --task encoder \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 128 \
  --concurrency 4 \
  --requests 500 \
  --warmup 50 \
  --arrival closed_loop \
  --output results/multistream_c4.json
```

模型必须是**只读 inference**:自定义 op、cache 或会写内部状态的模块,先看下方
"正确性校验"。

多进程 + MPS(在同一个 shell 里启动 MPS,确保环境变量传给后续 Python 进程):

```bash
source ./mps_control.sh start
python benchmark.py \
  --mode multiprocess \
  --model /path/to/model \
  --task encoder \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 128 \
  --concurrency 2 \
  --requests 500 \
  --warmup 50 \
  --arrival closed_loop \
  --output results/mps_p2.json
source ./mps_control.sh stop
```

每个进程独立加载模型:先估算显存,进程数上去后别 OOM。`CUDA_MPS_ACTIVE_
THREAD_PERCENTAGE` 是**上限**,不会为某个进程预留独占 SM(要在 start 前导出,
见 `mps_control.sh` 内注释)。

批量 sweep(一张 CSV 看全貌;MPS 已启动且显存够时加 `--include-multiprocess`):

```bash
python run_sweep.py \
  --model /path/to/model \
  --task encoder \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 128 \
  --requests 500 \
  --warmup 50 \
  --output-dir results
```

block slot 微基准(不依赖 Python 环境):

```bash
./build_and_run.sh              # 编译并依次跑 small / wide / smem 三组
./build_and_run.sh smem         # 只跑 128 threads + 48KB dynamic smem 一组
# 或手动:
nvcc -O3 -lineinfo block_slot_demo.cu -o block_slot_demo
./block_slot_demo 32 0 4096 200000      # 小 block:先撞 CTA/block slot 上限
./block_slot_demo 256 0 4096 200000     # 增大 block:warp slot 成为约束
./block_slot_demo 128 49152 4096 200000 # 加 dynamic smem:shared memory 成为约束
```

## 正确性校验:多 Stream 输出一致性

multistream 把一个模型实例同时暴露给 N 条 stream,而不少 Hugging Face 模型
假设"单线程串行 forward"。**正式测量前**先跑一轮一致性检查:

```bash
python benchmark.py --mode multistream --model /path/to/model --task encoder \
  --dtype bf16 --batch-size 1 --seq-len 128 --concurrency 4 --requests 50 \
  --warmup 20 --arrival closed_loop --consistency-check \
  --output results/consistency.json
```

`--consistency-check` 在测量结束后追加:先串行跑一次留参考输出,再 N 条 stream
以 barrier 对齐后**真正并发**各跑一次,逐张量对比,结果写入
`metadata.consistency_max_abs_error / consistency_max_rel_error`(非浮点输出
按不一致元素个数计入 max_abs)。只读推理应与串行结果一致(非确定性 kernel 会
带来小量浮点抖动);出现结构性偏差说明该模型不能进 multistream 模式。
multiprocess 模式每进程独立加载模型,不做此项检查。

## 到达模式

- `closed_loop`:固定最多 N 个 in-flight 请求,适合测容量上限;
- `burst`:一次性提交所有请求,适合观察 enqueue 与尾延迟放大;
- `poisson`:指数分布间隔,更接近随机到达,`--qps` 控制平均速率。

`multiprocess` 当前只支持 `closed_loop`。在线服务的真实 P99 还需要在服务框架、
RPC、batcher 和真实流量分布下复测。

## Profile 命令

Nsight Systems 看 kernel/memcpy 是否真重叠:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --output=results/multistream_c4 \
  python benchmark.py --mode multistream --model /path/to/model \
    --task encoder --dtype bf16 --batch-size 1 --seq-len 128 \
    --concurrency 4 --requests 100 --warmup 20 --arrival closed_loop
```

重点看:不同 stream 的 kernel overlap、launch gap、同步点、H2D/D2H overlap、
CPU submission 是否成为瓶颈。

Nsight Compute 缩短请求数,只 profile 少量 kernel:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --launch-skip 20 \
  --launch-count 10 \
  --export results/ncu_multistream \
  python benchmark.py --mode multistream --model /path/to/model \
    --task encoder --dtype bf16 --batch-size 1 --seq-len 128 \
    --concurrency 4 --requests 40 --warmup 10 --arrival closed_loop
```

重点看:Achieved Occupancy;SM Active / issue active;
`launch__occupancy_limit_blocks`、`launch__occupancy_limit_warps`;register /
shared-memory occupancy limit;Tensor Core 与主要 pipeline utilization;DRAM、
L2 throughput;warp stall reasons。block slot demo 同理:用 Nsight Systems 比较
两条 stream 是否重叠,用 Nsight Compute 确认限制来自 CTA slot、warp slot、
register 还是 shared memory。

## P99 Gate

不要用"GPU utilization 更高"作为胜出条件。每组配置先过硬门槛:

1. total P99 不超过业务 SLA;
2. 错误率、OOM、MPS hang 和结果正确性不退化;
3. 通过后再比较单卡吞吐、显存峰值和功耗。

建议至少重复 5 轮,报告中位数和最差一轮。若并发配置的 P99 对流量波形敏感,
单列 burst 结果,不要只给 closed-loop 数据。

## 已知边界

- `generate` 路径包含 autoregressive loop、KV cache 和框架内部同步,不能用
  encoder 结论直接外推;
- 某些 Hugging Face 模型或自定义 op 不是线程安全的;multistream 运行前先过
  上面的一致性检查;
- `torch.compile` 首轮包含编译开销,必须 warmup;动态 shape 会触发重编译或
  graph break;
- CUDA Graph 需要静态 shape、固定内存地址和受控 stream capture,本任务**没有**
  默认开启 Graph,避免把两个变量混进一个实验;
- 多进程会复制模型和 allocator 状态,MPS 不消除这部分显存成本。
