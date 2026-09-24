# references — 第三方经典资料存档

本目录收录与仓库主题相关的**第三方公开资料**（经典讲稿、论文等），方便对照
学习。这些资料**不是本仓库原创内容**，也不占用任务编号；版权归原作者及原
发布方所有，本仓库仅做学习交流用途的转载存档。

**声明：本目录所有资料仅供个人学习交流使用，请勿用于商业用途。版权归原作
者/原发布方所有；如版权方认为此处转载不妥，请通过 GitHub Issue 联系我，
我会第一时间删除。**

## 索引

| 文件 | 说明 | 来源 |
|---|---|---|
| [`Optimizing_Parallel_Reduction_in_CUDA_Mark_Harris.pdf`](Optimizing_Parallel_Reduction_in_CUDA_Mark_Harris.pdf) | CUDA 并行归约（reduce）优化的经典讲稿：从最朴素的树形归约出发，7 个 kernel 版本逐步消除分支发散、shared memory bank conflict、空闲线程与指令开销，最终逼近访存带宽上限（G80 上 2.08 GB/s → 62.7 GB/s，30 倍） | Mark Harris（NVIDIA Developer Technology），NVIDIA 公开讲稿，版权归 NVIDIA 所有 |
