# 任务 02：GEMM + ReduceScatter 的 peer write（epilogue 融合）

教程《GPU 计算与通信融合入门》第七章配套代码。属于"prologue/epilogue 融合"
路线：计算 CTA 在 GEMM epilogue 中直接把 partial result 写进 output owner GPU
的 rank slot，粒度是 CTA / tile。

数据流：

- 两张 GPU 分别保存 K 维的一半（A、B 的 K/2 分片），各自计算 partial C；
- kernel 内每个 thread 根据 output row 判断 owner（GPU0 还是 GPU1），直接把
  partial result peer write 到 owner GPU 上自己的 rank slot；
- 两个 rank 写入不同 slot，因此不要求跨 GPU atomic；owner 随后执行一次本地
  reduce，得到完整的 ReduceScatter 输出分片。

注意：GEMM 是朴素 O(MNK) 教学实现。生产版本会使用 Tensor Core tile、双缓冲、
vectorized remote write 和更细的 signal。

## 环境要求（独立于其他任务）

| 项 | 要求 |
|---|---|
| GPU | 同一节点 ≥2 张 NVIDIA GPU，互相支持 P2P（`nvidia-smi topo -m` 确认） |
| CUDA Toolkit | ≥11.4（需 `nvcc`，配套驱动） |
| 编译器 | nvcc 自带 host 编译链（gcc/clang 视平台） |
| 运行时依赖 | 无（纯 CUDA runtime，不需要 NCCL/NVSHMEM/torch） |

## 构建与运行

```bash
cd 02_fused_peer_write_rs
./build_and_run.sh                    # 编译到 ./build/ 并运行，参数透传
# 或手动：
nvcc -O2 -std=c++17 02_fused_peer_write_rs.cu -o build/fused_peer_write_rs
./build/fused_peer_write_rs 1024 1024 1024 2 5
```

参数：`M N K WARMUP_ITERS REPEAT_ITERS`。`M/N/K` 均需为正整数，`M/K` 为偶数
（2 卡各分一半）。后两个参数用于预热和重复采样，让 Nsight Systems 里能看到
多轮 `partial_gemm_peer_write` kernel 并发，降低首轮初始化噪声。

## 正确性校验

A、B 全 1，则输出每个元素都应等于 K，误差容忍 1e-5。打印 `PASS` 才算通过。
程序同时打印各卡的 partial GEMM 与本地 reduce 事件耗时。

## profiling

```bash
nsys profile -o report02 --force-overwrite true ./build/fused_peer_write_rs 512 512 512 2 5
nsys-ui report02.nsys-rep
```

观察要点：epilogue 远端写对 kernel 耗时的影响（对比去掉 peer write 的版本），
L2/PCIe/NVLink 流量（Nsight Compute）。
