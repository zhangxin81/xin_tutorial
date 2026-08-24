# 任务 04：copy engine near-zero-SM（硬件卸载）

教程《GPU 计算与通信融合入门》第九章配套代码。属于"硬件卸载"路线：
`cudaMemcpyPeerAsync` 负责 GPU0→GPU1 的 P2P copy，另一条 stream 在 GPU0 上运行
FMA kernel。若硬件和驱动选择 copy engine（DMA）数据路径，payload 搬运不需要
常驻通信 CTA，即 near-zero-SM 通信。

注意：程序会打印 `asyncEngineCount`，但该字段**不构成** overlap 保证；事件计时
只能给出"可能重叠"的提示（并发耗时明显小于串行耗时），最终仍要用 Nsight
Systems 看 memcpy 是否落在 copy engine lane。

## 环境要求（独立于其他任务）

| 项 | 要求 |
|---|---|
| GPU | 同一节点 ≥2 张 NVIDIA GPU，GPU0→GPU1 支持 P2P |
| CUDA Toolkit | ≥11.4（需 `nvcc`，配套驱动） |
| 运行时依赖 | 无（纯 CUDA runtime，不需要 NCCL/NVSHMEM/torch） |

## 构建与运行

```bash
cd 04_copy_engine_near_zero_sm
./build_and_run.sh              # 编译到 ./build/ 并运行，默认搬运 128 MiB
./build_and_run.sh 256          # 参数为 MiB 数，透传给二进制
# 或手动：
nvcc -O2 -std=c++17 04_copy_engine_near_zero_sm.cu -o build/copy_engine_near_zero_sm
./build/copy_engine_near_zero_sm
```

## 正确性校验

src 缓冲区置零，peer copy 后 dst 抽样应为 0；compute 输出必须有限。
打印 `PASS` 才算通过，并打印 compute/copy/wall/串行四个耗时。

## profiling

```bash
nsys profile -o report04 --force-overwrite true ./build/copy_engine_near_zero_sm
nsys-ui report04.nsys-rep
```

观察要点：memcpy 行是否与 SM kernel 并发、是否落在 copy engine lane，
而不是变成 SM 上的 copy kernel。
