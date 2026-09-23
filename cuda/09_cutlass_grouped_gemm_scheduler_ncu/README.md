# 任务 09:torch 原生 grouped GEMM 为什么比"手写 CUTLASS"快 25%

教程《torch 凭什么比手写 CUTLASS 快 25%?》(发布后补链接)配套代码。业务形状
是 MoE 专家投影:10 个专家、每个 128x2048x1024 的 BF16 grouped GEMM(FP32
累加)。自建算子库按"大矩阵经验"配了 CUTLASS Cooperative 调度 + 128x256x64
大 tile,稳态计时却比 torch 原生 `_grouped_mm` 慢 25%;用 NCU 打开一看,torch
调的也是 CUTLASS——只是选了 Pingpong 调度 + 64x128x128 小 tile。

一句话:**差距不在"有没有用 CUTLASS",而在"用 CUTLASS 的哪张牌";本任务用
计时、kernel 名解码、SASS 指令与 NCU section 四路证据把这张牌的选法讲清。**

## 核心数字速查

稳态 CUDA-event 计时(预热后 3000 次平均;2026-09-11,H100 80GB HBM3、
CUDA 12.9、PyTorch 2.9.1+cu129、NCU 2025.2.1):

| 路径 | 调度族 / tile | 耗时 | 吞吐 | 相对 |
|---|---|---:|---:|---:|
| torch `_grouped_mm` | Pingpong,64x128x128 | 0.0319 ms | 168.1 TFLOP/s | 1.00x |
| CUTLASS 基线 | Cooperative,128x256x64 | 0.0400 ms | 134.1 TFLOP/s | 0.80x |
| CUTLASS 调优 | Pingpong,64x128x128 | 0.0321 ms | 167.1 TFLOP/s | 0.99x |

NCU 证据(精简指标与全 section 两次采集,复现命令见下文 Profile 一节):

- **tile 账**:128x256 tile 切出 10x8=80 个输出 tile,H100 有 132 个 SM,
  52 个 SM(39%)全程空转;64x128 tile 切出 320 个,每个 SM 分到 2~3 个。
- **寄存器账**:三个 kernel 都是 168 regs/thread(384x168=64512,恰好卡进
  每 SM 64K 寄存器堆,NCU `occupancy_limit_registers=1`)。差别在累加器预算:
  Cooperative 两组合算 128x256,每组 64x256,累加器 128 regs/thread;Pingpong
  每组独占整个 tile,64x128 只要 64 regs/thread;若给 Pingpong 配 128x256,
  累加器需要 256 regs/thread,超过 255 的硬件上限必炸——这就是"大 tile +
  FP32 累加只能 Cooperative"的来历。
- **SASS 证据**:三 kernel 都是 4 级 mainloop
  流水;主循环为 `HGMMA.64x128x16/64x256x16.F32.BF16`(SS 源);搬运为
  `UTMALDG.2D`(cluster 2x1x1 下 B 走 `UTMALDG.2D.MULTICAST`);Warp
  Specialization 的寄存器再平衡直接可见:消费者 `USETMAXREG.TRY_ALLOC 232`,
  生产者 `USETMAXREG.DEALLOC 40`。
- **阻塞构成**(完整 section 采集,每条发射指令的平均等待周期):torch/Pingpong
  第一大阻塞是 Long Scoreboard(10.8/18.2 周期,59.3%,等 L2/显存);基线
  Cooperative 是 Barrier Stall(5.6/17.5 周期,31.7%,等 CTA 内兄弟
  warpgroup 齐步)——"两组共同进退"的代价直接可见;调优 Pingpong 回到
  Long Scoreboard 主导(9.0/18.7 周期,48.2%)。

## 选型规则(本任务验证过的版本)

1. 对齐 16B 满足就用 TMA 系调度族,不满足退回 CpAsync。
2. 先数 tile 数:输出 tile 数 = Σ 每组 ⌈M/tileM⌉x⌈N/tileN⌉。不满一个 wave
   (132)时 epilogue 暴露不可避免,先用小 tile 把机器填满(每 SM 2~3 个 tile
   最舒服)。
3. 再算寄存器账:Pingpong 累加器开销 = tileMxtileN/128 regs/thread,必须
   ≤255 且留余量;128x256 + FP32 累加只能 Cooperative。
4. 形状碎(MoE、变长)选小 tile + Pingpong;又大又方的单矩阵可放心
   Cooperative + 大 tile;之后再用 CTA swizzle 微调 L2 复用。

## 文件

| 文件 | 内容 |
|---|---|
| `src/cutlass_grouped_gemm.cu` | grouped GEMM 基准,基于 NVIDIA CUTLASS 示例 57(BSD-3,文件头保留原始许可与出处),本地追加了机器可读的 `RESULT` 输出行,计算路径未改;`--schedule=coop|pingpong` 切换调度族 |
| `src/profile_grouped_mm.py` | torch `_grouped_mm` 稳态计时基准(CUDA events;等价复刻本任务所用脚本口径) |
| `build_and_run.sh` | 一键构建 + case 形状双调度冒烟 |
| `scripts/build.sh` | `nvcc -O3 -std=c++17 -arch=sm_90a` 编译,需 `CUTLASS_DIR` 指向 CUTLASS 3.x 检出 |
| `scripts/profile_ncu.sh` | 三路径 NCU 采集(5 个头条指标;`FULL=1` 改为 `--set full` 全 section),导出 ncu-rep / raw csv / sass |
| `scripts/parse_ncu_report.py` | 离线解析采集导出的 CSV:调度族、tile、stage、时长、利用率 |

所有 profiling 产物(ncu-rep、raw csv、sass 导出)按仓库规范写在仓库外的
`../../worker_results/09_cutlass_grouped_gemm_scheduler_ncu/`,不入库;可用
`FULL=1 scripts/profile_ncu.sh` 重新生成全 section 报告,再用
`scripts/parse_ncu_report.py` 离线读数。

## 环境要求

| 项 | 要求 |
|---|---|
| GPU | 1 张 H100(SM90);Cooperative 要求 TileShape_M>=128 的约束在本任务形状下不触发 |
| CUDA Toolkit | >=12.3(`nvcc`) |
| CUTLASS | 3.x 检出(环境变量 `CUTLASS_DIR`) |
| Nsight Compute | >=2025.x CLI,需要性能计数器权限 |
| Python + PyTorch | torch 路径需带 Hopper grouped GEMM 支持的 PyTorch(>=2.x,`torch._grouped_mm`) |

## 构建与运行

```bash
cd cuda/09_cutlass_grouped_gemm_scheduler_ncu
export CUTLASS_DIR=/path/to/cutlass
./build_and_run.sh                                    # 构建 + 双调度冒烟

# 与 torch 对比(稳态口径)
python3 src/profile_grouped_mm.py --groups 10 --m 128 --k 1024 --n 2048 --warmup 80 --repeat 3000
./build/cutlass_grouped_gemm --schedule=coop     --groups=10 --m=128 --n=2048 --k=1024 --alpha=1 --beta=0 --iterations=5000 --no-verify
./build/cutlass_grouped_gemm --schedule=pingpong --groups=10 --m=128 --n=2048 --k=1024 --alpha=1 --beta=0 --iterations=5000 --no-verify
```

## Profile 命令

```bash
scripts/profile_ncu.sh                       # 5 指标精简采集 -> ../../worker_results/.../raw_csv/
FULL=1 scripts/profile_ncu.sh                # 全 section 采集(replay 次数多)
python3 scripts/parse_ncu_report.py          # 默认读上一行的 CSV 目录,也可显式传路径
rg -n 'HGMMA|UTMALDG|USETMAXREG' ../../worker_results/09_cutlass_grouped_gemm_scheduler_ncu/sass_*.txt
```

## 边界说明

- `torch._grouped_mm` 的调度选择是 PyTorch 内部实现,随版本变化;本任务记录
  的是 PyTorch 2.9.1+cu129 的行为。
- cooperative/pingpong 的适用规则来自本任务实测与 NVIDIA 官方讲座口径,边界
  情形(K 极深、FP16 累加、超大 tile)请按第 2/3 条规则自行复算。
- 本文数值仅代表 2026-09-11 那次采集(特定 GPU、驱动、库版本与形状),
  不代表 H100 的统一成绩。
