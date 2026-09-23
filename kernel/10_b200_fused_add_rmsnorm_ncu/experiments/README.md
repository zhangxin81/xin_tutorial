# experiments/：实验目录与脚本

这里放教程正文展开的 4 个实验目录，目录编号 01–04。根目录的 `kernel.cu` / `binding.cpp` / `submission.py` 是最终
保留的完整实现，不在本目录内。

## 目录

| 目录/文件 | 内容与结论 |
|---|---|
| `01_triton_w16_maxnreg32/` | Triton，`num_warps=16` + `maxnreg=32`：寄存器 34→32，occupancy 64.92%→86.53%，无 spill，duration 42.08→40.35 us |
| `02_cute_dsl_copy128/` | CUTE DSL，显式 128-bit copy atom：与 01 lowering 后硬件形态几乎一致 |
| `03_cuda_graph_replay/` | CUDA C++ + 按 rows 缓存 Graph Exec：cold-L2 smoke 中 1.0188× vs 02；根实现的 Graph 入口由它演化而来 |
| `04_cuda_tma_bulk_smem/` | TMA staging 负例：两个 8192 行 workload 约 0.082 ms vs 直接路径 0.066 ms，shared 中转不划算 |
| 根目录实现 | 完整 CUDA 实现（512/1024 线程 dispatch + Graph 缓存，两条累加链）。cold-L2 smoke 中 1.0073× vs 02，落后同轮的 03；它被保留是因为稳定、可读，而不是因为它是实测最快 |

## 测量协议

benchmark 统一用 `score_aligned_benchmark.py`：16 个官方 workload 全部先
过 correctness（逐元素与 `reference.py` 比对），默认清空 L2、预热后多轮
重复，按各 workload 中位数汇总，并以同一轮内的 anchor 候选校准跨轮噪声。
教程正文引用的同轮对照即此协议的 warmup 3 / 重复 20 / 3 轮档。

```bash
# 三候选冒烟（编译根实现 + official/01 对比）
./build_and_run.sh
# 全量协议
python3 experiments/score_aligned_benchmark.py --configs official,triton01,cute02,cuda03,cuda04,root_current
# 同轮 smoke 对照（教程正文 03/04 与根实现的结论口径）
python3 experiments/score_aligned_benchmark.py --configs cute02,cuda03,cuda04,root_current --warmup 3 --repeats 20 --rounds 3
# NCU 采集（rows=8192 与 131，输出在 experiments/ncu_out/，不入库）
NCU_BIN=/path/to/ncu ./experiments/run_ncu.sh
```

`--set full` 会触发 Kernel Replay（一次完整采集约 39 passes），报告里的
duration 是 profiler 环境的数字，不能当 benchmark 用；CUDA Graph 的 host
侧收益 NCU 单 kernel 时长测不到，须回到 CUDA event / 全量基准。
