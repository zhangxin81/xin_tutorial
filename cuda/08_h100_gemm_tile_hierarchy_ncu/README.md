# 任务 08:用 Nsight Compute 读出 H100 GEMM 的三级分块

教程《矩阵乘法在 H100 上是怎么分块计算的》配套代码。示例程序不做任何手写
kernel 优化:它只是调用 cuBLASLt 算一个 8192³ BF16→FP32 矩阵乘法,然后用
Nsight Compute 抓住库选中的那条 kernel(`nvjet_tss_320x128_64x3_1x2_h_bz_
coopB_NNT`),从 kernel 名字、Launch Statistics、SASS 指令和流量指标四路证据
里,把 threadblock tile / warp tile / thread tile 三级分块结构拼出来。

一句话:**tile 层次不是文档里抄来的,是从报告里算出来的。**

## 文件

| 文件 | 内容 |
|---|---|
| `src/h100_cublaslt_gemm.cu` | cuBLASLt BF16 GEMM 基准:填充输入、启发式选算法、CUDA events 计时、`--verify` 打印 checksum |
| `build_and_run.sh` | 一键构建 + 4096 阶冒烟运行 |
| `scripts/build.sh` | `nvcc -O3 -std=c++17 -arch=sm_90` 链接 cuBLASLt |
| `scripts/profile_ncu.sh` | NCU 采集一发 kernel(`--launch-skip 20 --launch-count 1 --set full`)并导出 raw.csv / sass.txt / ncu_summary.md |
| `scripts/parse_ncu_report.py` | 从导出文件里解析 TMA / Tensor Core 指令证据与常用指标 |
| `scripts/verify_ncu_evidence.sh` | 证据门禁:报告里必须同时出现 TMA 与 Tensor Core 指令及其计数 |
| `reports/` | 一次已验证采集(2026-09-09,H100 80GB HBM3,NCU 2025.2.1)的证据文件,教程中所有数字的出处 |

`reports/` 下 `raw.csv` 是原始指标宽表,`ncu_summary.md` 是解析摘要,
`sass_excerpt.txt` 是 SASS 关键区域摘录(mbarrier 初始化、HGMMA 主循环、
TMA store 尾声、TMA load 搬运区)。完整 `sass.txt`(约 800 KB)由
`profile_ncu.sh` 重新生成,不入库。`reports/archive/` 存每次采集的时间戳
备份,亦不入库。

## 核心证据速查

采集对象:8192×8192×8192,BF16 输入、FP32 累加输出,列主序,α=1、β=0。

**kernel 名字段**(库内部命名,只作线索,不是 API 承诺):

| 字段 | 含义 | 置信度 |
|---|---|---|
| `320x128` | CTA/threadblock 输出 tile | 高(与 smem、寄存器、流量三笔账互相印证) |
| `64x3` | K 方向每阶段 64、3 级流水 | 高(同上) |
| `h` | 半精度家族 | 高(SASS 实测 BF16) |
| `coopB` | B 操作数协作/多播加载 | 中(`UTMALDG.3D.MULTICAST` + cluster=2 支持) |
| `nvjet/tss/1x2/bz/NNT` | 库内部调度、分解与布局标签 | 低(仅用于对比两个 kernel 名) |

**三笔账**(报告值 vs 按名字推导的 tile 参数复算):

| 账 | 报告值 | 推导值 |
|---|---|---|
| 共享内存 | 188636 B/CTA | (320×64 + 64×128)×2 B×3 级 = 172032 B,加 barrier/对齐余量 |
| 寄存器 | 168/thread | 64×64 FP32 累加器 ÷128 线程 = 32;一个 warpgroup 5 个累加器 = 160,+8 杂项 |
| TMA 加载流量 | 12.2138 GB | 1664 tile×128 K 步×57344 B ≈ 12.21 GB |

SASS 证据:主循环是两个代码区域 × 5 组累加器(R24/R56/R88/R120/R152)×
4 条 `HGMMA.64x64x16.F32.BF16`,每组后跟 `WARPGROUP.DEPBAR.LE`;搬运区是
`UTMALDG.3D.MULTICAST`(B 多播,cluster=2)与 `UTMALDG.3D`;尾声用
`UTMASTG.3D` + `UTMACMDFLUSH` 写回,store 流量 268435456 B 恰好等于
8192²×4 的完整输出。10 组 64×64 累加器拼成 320×128 的 CTA tile,即
CTA tile = 2 warpgroup × 5 warp tile(64×64),thread tile = 每线程 32 个
FP32 累加寄存器。

## 环境要求(独立于其他任务)

| 项 | 要求 |
|---|---|
| GPU | 1 张 H100(SM90);其他架构可跑但 TMA/HGMMA 证据会不同 |
| CUDA Toolkit | ≥12.x(`nvcc` 与 cuBLASLt) |
| Nsight Compute | ≥2025.x CLI(`ncu`),需要性能计数器权限(Linux root 或 `CAP_SYS_ADMIN`/perf_event_paranoid 配置) |
| Python | ≥3.8(仅解析脚本用,标准库) |

## 构建与运行

```bash
cd cuda/08_h100_gemm_tile_hierarchy_ncu
./build_and_run.sh                          # 构建 + 4096 阶冒烟
./build/h100_cublaslt_gemm --m=8192 --n=8192 --k=8192 --warmup=20 --iters=200
```

注意:`--verify` 只打印前 16 个输出元素之和,不是完整正确性校验;要做严格
校验,应在小矩阵上与 CPU 参考逐元素比对。

## Profile 命令

```bash
scripts/profile_ncu.sh --m=8192 --n=8192 --k=8192 --warmup=20 --iters=200
scripts/verify_ncu_evidence.sh
rg -n 'HGMMA|WARPGROUP|UTMALDG|UTMASTG|UTMACMDFLUSH' reports/sass.txt
```

`--set full` 需要 39 次 replay pass,剖析时间明显长于正常运行;报告里的
1.578 ms 是被剖析那一发 kernel 的时间,程序自计时的 1.504 ms(avg)来自
CUDA events,两者口径不同,不能混在一组数据里比较。没有 H100 时,可以用
NCU GUI 直接打开 `reports/` 下这份证据对应的 `.ncu-rep`(见教程附件),
或离线解析 `raw.csv` / `sass_excerpt.txt`。

## 边界说明

- kernel 名与调度策略是 cuBLASLt 内部实现,随 CUDA 版本变化;本任务方法
  (名字→launch→SASS→流量四路交叉验证)不依赖特定版本。
- 1664 个逻辑 tile 对 132 个 CTA,说明 CTA 会串行处理多个 tile;精确的
  tile 调度算法是闭源实现,本任务只证明"一 CTA 多 tile",不还原调度器。
- `reports/` 数值仅代表该次采集(特定 GPU、驱动、库版本与形状),不代表
  H100 的统一成绩。
