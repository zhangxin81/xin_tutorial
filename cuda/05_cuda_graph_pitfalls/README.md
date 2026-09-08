# 任务 05:CUDA Graph 三条硬约束的易错场景与修复

教程《CUDA Graph:一次录制、多次重放》配套代码。capture 把一串 kernel 的调用
顺序、参数、地址录成一张图,之后每步一次 `cudaGraphLaunch` 提交,replay 之间
不再有逐 kernel 的 CPU launch——这是"关掉 CUDA Graph 后性能明显下降"的原因
(`perf` 子命令给出 C++ 层的量化对比;Python 框架里逐 kernel 的 dispatch 开销
更大,收益也更夸张)。

代价是三条硬约束。本任务把它们各造成一个**可直接复现的 bug**,再给出修复:

1. **形状、地址、参数在录制时固定,replay 不重算。** 精确说法:固化的是
   "参数值和地址",不是"数据"——指针指向的显存内容每次 replay 都是现读的。
   官方逃生口:`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate`
   (参数、指针、gridDim/blockDim 都能改)、参数改走 device 内存、按形状分桶
   重新 capture 或 pad 到固定形状。
2. **只有提交到 capture stream 的 GPU 操作会进图;CPU 侧逻辑只在 capture 那一刻
   执行一次。** capture 区域里调 `cudaStreamSynchronize` / `cudaEventSynchronize`
   / 同步版 `cudaMemcpy` 会直接作废 capture;把 CPU 逻辑"顺手"写进 capture 区域
   则是安静失败——不报错,但 replay 时不会重新执行。`cudaLaunchHostFunc`
   (host node)可以把 CPU 回调塞进图,但回调里不许调 CUDA API,且会阻塞所在
   支路,别当通用手段。
3. **图执行期间没有 CPU 参与,kernel 背靠背。** 图只保留你 capture 时显式表达的
   依赖边:原来靠 CPU 提交延迟"碰巧"掩盖的 race 会现形;replay 中间也没有 CPU
   插手检查/决策的点,要么把检查放在 replay 之间,要么拆图。

## 环境要求(独立于其他任务)

| 项 | 要求 |
|---|---|
| GPU | ≥1 张 NVIDIA GPU(全部 demo 单 GPU,不需要 P2P/NCCL/NVSHMEM/torch) |
| CUDA Toolkit | ≥11.4(`nvcc`,三参版 `cudaGraphInstantiate`;实测 12.9) |
| 运行时依赖 | 无 |

## 构建与运行

```bash
cd 05_cuda_graph_pitfalls
./build_and_run.sh                # 编译并按顺序跑全部 8 个 demo
./build_and_run.sh s1bad          # 单跑某个子命令
# 或手动:
nvcc -O2 -std=c++17 05_cuda_graph_pitfalls.cu -o build/cuda_graph_pitfalls
./build/cuda_graph_pitfalls all
```

| 子命令 | 演示内容 |
|---|---|
| `perf` | 16 个短 kernel × 500 步:逐 kernel 提交 vs 一次 replay 的 CPU 开销 |
| `s1bad` | by-value 参数 + 输出指针固化:改宿主变量再 replay,输出仍用旧值进旧地址 |
| `s1fix_data` | 修复 A:参数走 device 内存,图内 H2D 节点从固定地址 pinned 缓冲现读 |
| `s1fix_update` | 修复 B:`cudaGraphExecKernelNodeSetParams` 每步改参数值和输出地址 |
| `s2bad` | capture 里放同步(大声失败:capture 作废);CPU 数据准备进图(安静失败:只执行一次) |
| `s2fix` | 修复:CPU 每步的活在图外干,固定地址 pinned 做图的 I/O 口,同步放 replay 之间 |
| `s3bad` | 两条 stream 无显式依赖:靠 CPU 提交间隙碰巧没事,进图后 race 现形 |
| `s3fix` | 修复:显式 event 边写进图,背靠背但有序 |

运行约十几秒。退出码 0 = 所有 demo 与上述描述一致(bad demo 的成功标准是
bug 被复现)。

## 场景速览(现象 → 原因 → 修复)

- **s1**:现象——宿主把 `scale` 从 2 改成 5、把输出指针换到新缓冲,replay 后
  新缓冲无人动、旧缓冲按 `scale=2` 被重写。原因——kernel 参数在 capture 时按值
  烤进图,指针同理。修复——参数放 device 内存换"数据"(s1fix_data),或用
  exec 更新 API 换"参数和地址"(s1fix_update)。
- **s2**:现象(a)——capture 区域里一调 `cudaStreamSynchronize`,capture 立即
  作废,后续提交和 EndCapture 全部报错;现象(b)——CPU 填数据的循环写进
  capture,不报错,但 replay 每步拿到的都是 capture 时刻那份输入。修复——
  CPU 准备/检查放在 replay 之间(图外),图通过固定地址 pinned 缓冲收发数据
  (s2fix,即推理框架"静态输入/输出缓冲"的姿势)。
- **s3**:现象——写者/读者在两条 stream、无 event 依赖;非图模式靠 1ms 的
  CPU 提交间隙碰巧不出事,capture 进同一张图后两个节点同时起跑,读者稳稳读到
  旧值。修复——把真实依赖用 event 显式写进图(s3fix):仍是一次提交、背靠背,
  但顺序有保证。图没有制造 bug,只是撤掉了 CPU 提交延迟这块挡板。

## 另外两个常见坑(代码未演示,只提示)

- `cudaMalloc` 不许出现在 capture 区域(分配要用图 API 的 memory node,或
  memory pool 的 `cudaMallocAsync` 再 capture,CUDA ≥11.4)。实践:capture 前
  把显存分配好,并保证 replay 期间分配地址稳定——图里的地址是裸地址,`free`
  后继续 replay 是悬垂写。
- 动态形状:gridDim/blockDim 同样在录制时固化。按形状分桶各 capture 一张图,
  或 pad 到固定形状(推理框架按 batch 桶预 capture 的做法);
  `cudaGraphExecKernelNodeSetParams` 也能改 launch 配置,但注意它的约束与开销。

## profiling

```bash
nsys profile -o report05 --force-overwrite true ./build/cuda_graph_pitfalls all
nsys-ui report05.nsys-rep
```

观察要点:非图段一串短 kernel 之间存在 launch gap(CPU 在两个 kernel 之间才
提交下一个),`perf` 的图段里同一串 kernel 背靠背、间隔消失;CUDA API 轨道上
一次 replay 对应一次 `cudaGraphLaunch`。注意 nsys 默认把整张图聚合成一条
`CUDA Graph Trace` 记录,要看图内逐 kernel 时间线需加
`--cuda-graph-trace=node`。

## 实测参考(2026-09-02,H100 80GB,CUDA 12.9,`--cuda-graph-trace=node`)

- perf:非图 42.2 us/step(16 次 launch,CPU 提交瓶颈),图 1.2 us/step
  (1 次 replay);时间线上相邻 kernel 间隔中位数:非图 1.95 us vs 图内
  0.096 us——背靠背直接可见。`cudaGraphLaunch` 次数与代码严格一致。
- s1bad:输出按旧参数写进旧地址,新地址哨兵未被触碰;s1fix 两种修复每步数值
  均正确。图内 memcpy 节点(H2D 8B 参数、4KB 数据)在时间线可见。
- s2bad(a):`cudaStreamSynchronize` 返回 "operation not permitted when stream
  is capturing",作废后的提交与 EndCapture 均报 "previous error during
  capture",graph=nullptr;(b) 三步 replay 全部停在 capture 时刻的 101。
- s3bad:非图 + 1ms 提交间隙,旧值 0/1048576(碰巧没事);无间隙裸跑与进图
  replay 均为 1048576/1048576,时间线确认写/读两 kernel 并发约 148 us;
  s3fix 显式 event 边后完全串行,间隔 0.3 us——背靠背但有序。
- 完整日志与 nsys 报告:`../../../worker_results/run_20260902_05_cuda_graph/`。
