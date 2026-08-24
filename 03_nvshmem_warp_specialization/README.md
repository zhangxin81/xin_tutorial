# 任务 03：NVSHMEM 单 kernel 内 warp 分工（warp specialization）

教程《GPU 计算与通信融合入门》第八章配套代码。属于"kernel 内 warp/CTA 分工"
路线：通信与计算处于同一个 CUDA kernel，由不同 warp 承担，硬件调度允许时可以
重叠。

分工：

- kernel 启动 8 个 warp：warp 0 的 lane 0 对下一 PE 发起 `nvshmem_float_put`
  （单线程 bulk put）；
- 其余 7 个 warp 做与通信无依赖的本地 FMA 计算。

教学代码使用单线程 bulk put，是为了把 producer/consumer 角色看清。NVSHMEM
官方同时提供 block 级 put 等 thread-group API，适合让整个 CTA 协作搬运。
生产实现通常加入双缓冲、ready signal、tile 状态机，并避免 producer warp
长时间阻塞。

## 环境要求（独立于其他任务）

| 项 | 要求 |
|---|---|
| GPU | 同一节点 ≥2 张 NVIDIA GPU，互相支持 P2P |
| CUDA Toolkit | ≥11.4（需 `nvcc`，配套驱动） |
| NVSHMEM | ≥2.x（[官网下载](https://developer.nvidia.com/nvshmem)），需设置 `NVSHMEM_HOME` |
| 运行时依赖 | 无 torch/NCCL |

本任务是四个任务中唯一有第三方库依赖的，其余任务无需 NVSHMEM。

## 安装与配置

```bash
# 下载并解压 NVSHMEM 后：
export NVSHMEM_HOME=/path/to/nvshmem       # 其下应有 include/ lib/ bin/
```

## 构建与运行

```bash
cd 03_nvshmem_warp_specialization
./build_and_run.sh                      # 编译到 ./build/ 并用 nvshmrun -np 2 运行
./build_and_run.sh 4194304 256          # 参数透传：N（元素数）、计算迭代数
# 进程数可通过环境变量覆盖：
NPES=4 ./build_and_run.sh
```

## 正确性校验

每个 PE 的 recv 缓冲区应等于前一个 PE 的编号（环形 put），compute 输出必须
有限。每个 PE 打印 `PASS` 才算通过，附带 kernel+barrier 事件耗时。

## profiling

```bash
nsys profile -o report03 --force-overwrite true \
    ${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./build/nvshmem_warp_specialization
nsys-ui report03.nsys-rep
```

观察要点：producer warp 的 put 与其余 warp 的 FMA 是否在同一 kernel 的
timeline 内并发。
