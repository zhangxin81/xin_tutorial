// 示例三：NVSHMEM 单 kernel 内 warp 分工（warp specialization）。
//
// 对应文章《GPU 计算与通信融合入门》第八章。属于“kernel 内 warp/CTA 分工”
// 路线：通信与计算处于同一个 CUDA kernel，由不同 warp 承担，硬件调度允许时
// 可以重叠。
//
// 分工：
//   - 这个 kernel 启动 8 个 warp：warp 0 的 lane 0 对下一 PE 发起
//     nvshmem_float_put（单线程 bulk put）；
//   - 其余 7 个 warp 做与通信无依赖的本地 FMA 计算。
//
// 说明：教学代码使用单线程 bulk put，是为了把 producer/consumer 角色看清。
// NVSHMEM 官方同时提供 block 级 put 等 thread-group API，适合让整个 CTA
// 协作搬运。生产实现通常加入双缓冲、ready signal、tile 状态机，并避免
// producer warp 长时间阻塞。
//
// 运行：
//   export NVSHMEM_HOME=/path/to/nvshmem
//   nvcc -O2 -std=c++17 -rdc=true \
//       -I${NVSHMEM_HOME}/include 03_nvshmem_warp_specialization.cu \
//       -L${NVSHMEM_HOME}/lib -lnvshmem_host -lnvshmem_device \
//       -o nvshmem_warp_specialization
//   ${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./nvshmem_warp_specialization
//   # 自定义规模：
//   nvshmrun -np 2 ./nvshmem_warp_specialization 4194304 256
//
// profiling：
//   nsys profile -o 03_nvshmem --force-overwrite true \
//       ${NVSHMEM_HOME}/bin/nvshmrun -np 2 ./nvshmem_warp_specialization
//
// 硬件要求：同一台机器上至少 2 张支持 P2P 的 NVIDIA GPU + NVSHMEM 库。

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

// 教学版 warp specialization：
// - warp 0 的 lane 0 发起到下一 PE 的 bulk put；
// - 其余 warp 执行与通信无依赖的本地计算。
// 两类工作处于同一个 CUDA kernel，可以在硬件调度允许时重叠。
__global__ void comm_compute_kernel(
    const float* symmetric_send,
    float* symmetric_recv,
    float* local_compute,
    int n, int mype, int npes, int compute_iters) {
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  if (warp == 0) {
    if (lane == 0) {
      const int peer = (mype + 1) % npes;
      nvshmem_float_put(symmetric_recv, symmetric_send, n, peer);
      nvshmem_quiet();  // 确保本 PE 发起的 put 远端完成后再退出 producer 路径。
    }
  } else {
    const int compute_tid = (warp - 1) * warpSize + lane;
    const int compute_threads = (blockDim.x / warpSize - 1) * warpSize;
    for (int i = compute_tid; i < n; i += compute_threads) {
      float x = static_cast<float>(i + 1);
      // 多做一些独立计算，让 profiler 更容易看到 overlap。
      for (int r = 0; r < compute_iters; ++r) x = fmaf(x, 1.00001f, 0.00001f);
      local_compute[i] = x;
    }
  }
}

// 用法: ./nvshmem_warp_specialization [N] [COMPUTE_ITERS]
// 默认 N = 1<<20（4 MiB float），COMPUTE_ITERS = 128。
static void parse_args(int argc, char** argv, int* n, int* compute_iters) {
  *n = 1 << 20;
  *compute_iters = 128;
  if (argc >= 2) *n = std::atoi(argv[1]);
  if (argc >= 3) *compute_iters = std::atoi(argv[2]);
  if (*n <= 0 || *compute_iters <= 0) {
    std::fprintf(stderr, "要求 N、COMPUTE_ITERS 为正整数\n");
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  int N = 0, compute_iters = 0;
  parse_args(argc, argv, &N, &compute_iters);

  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (npes < 2) {
    if (mype == 0) std::fprintf(stderr, "需要至少两个 NVSHMEM PE\n");
    nvshmem_finalize();
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(local_pe));
  const size_t bytes = static_cast<size_t>(N) * sizeof(float);

  // 每个 PE 必须按相同顺序、相同大小分配 symmetric objects。
  float* send = static_cast<float*>(nvshmem_malloc(bytes));
  float* recv = static_cast<float*>(nvshmem_malloc(bytes));
  float* compute = nullptr;
  CUDA_CHECK(cudaMalloc(&compute, bytes));
  if (!send || !recv) {
    std::fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
    nvshmem_global_exit(3);
  }

  std::vector<float> host_send(N, static_cast<float>(mype));
  CUDA_CHECK(cudaMemcpy(send, host_send.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(recv, 0, bytes));
  CUDA_CHECK(cudaMemset(compute, 0, bytes));
  nvshmem_barrier_all();

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, stream));
  comm_compute_kernel<<<1, 256, 0, stream>>>(
      send, recv, compute, N, mype, npes, compute_iters);
  CUDA_CHECK(cudaGetLastError());

  // 等待所有 PE 的 kernel/远端写结束，再读取 recv。
  nvshmemx_barrier_all_on_stream(stream);
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, end));

  std::vector<float> host_recv(N);
  std::vector<float> host_compute(N);
  CUDA_CHECK(cudaMemcpy(host_recv.data(), recv, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_compute.data(), compute, bytes, cudaMemcpyDeviceToHost));

  const int previous = (mype - 1 + npes) % npes;
  bool ok = true;
  for (int i = 0; i < N; ++i) {
    if (host_recv[i] != static_cast<float>(previous) || !std::isfinite(host_compute[i])) {
      ok = false;
      std::fprintf(stderr, "PE %d mismatch at %d: recv=%f expected=%d compute=%f\n",
                   mype, i, host_recv[i], previous, host_compute[i]);
      break;
    }
  }

  std::printf("PE %d/%d: %s (N=%d, compute_iters=%d, kernel+barrier=%.3f ms)\n",
              mype, npes, ok ? "PASS" : "FAIL", N, compute_iters, kernel_ms);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);
  cudaFree(compute);
  nvshmem_free(send);
  nvshmem_free(recv);
  nvshmem_finalize();
  return ok ? 0 : 1;
}
