// 示例四：copy engine 数据路径的 near-zero-SM 重叠。
//
// 对应文章《GPU 计算与通信融合入门》第九章。属于“硬件卸载”路线：
// cudaMemcpyPeerAsync 负责 GPU0→GPU1 的 P2P copy，另一条 stream 在 GPU0 上
// 运行 FMA kernel。如果硬件和驱动选择 copy engine（DMA）数据路径，payload
// 搬运不需要常驻通信 CTA，即 near-zero-SM 通信。
//
// 注意：程序会打印 asyncEngineCount，但该字段不构成 overlap 保证；两条
// stream 的事件计时只能给出“可能重叠”的提示（并发耗时明显小于串行耗时），
// 最终仍要用 Nsight Systems 看 timeline 上 memcpy 是否落在 copy engine lane。
//
// 运行（本目录自带独立环境说明，见 README.md；一键脚本 ./build_and_run.sh）：
//   ./build_and_run.sh                    # 默认搬运 128 MiB
//   ./build_and_run.sh 256                # 自定义 MiB 数
// 或手动：
//   nvcc -O2 -std=c++17 04_copy_engine_near_zero_sm.cu -o build/copy_engine_near_zero_sm
//   ./build/copy_engine_near_zero_sm
//
// profiling：
//   nsys profile -o 04_copy_engine --force-overwrite true ./build/copy_engine_near_zero_sm
//   nsys-ui 04_copy_engine.nsys-rep       # 看 memcpy 行是否与 SM kernel 并发
//
// 硬件要求：同一台机器上至少 2 张支持 P2P 的 NVIDIA GPU。

#include <cuda_runtime.h>
#include <algorithm>
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

__global__ void busy_compute(float* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = static_cast<float>(i % 1024) * 0.001f + 1.0f;
  for (int r = 0; r < 128; ++r) x = fmaf(x, 1.000001f, 0.000001f);
  out[i] = x;
}

// 用法: ./copy_engine_near_zero_sm [MIB]，默认 128。
static int parse_mib(int argc, char** argv) {
  int mib = 128;
  if (argc >= 2) mib = std::atoi(argv[1]);
  if (mib <= 0) {
    std::fprintf(stderr, "要求 MIB 为正整数\n");
    std::exit(2);
  }
  return mib;
}

int main(int argc, char** argv) {
  const int mib = parse_mib(argc, argv);

  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count < 2) {
    std::fprintf(stderr, "需要至少两张 GPU\n");
    return 2;
  }

  int can = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
  if (!can) {
    std::fprintf(stderr, "GPU 0 -> GPU 1 不支持 P2P\n");
    return 2;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU0=%s asyncEngineCount=%d concurrentKernels=%d\n",
              prop.name, prop.asyncEngineCount, prop.concurrentKernels);

  CUDA_CHECK(cudaSetDevice(0));
  cudaError_t peer_e = cudaDeviceEnablePeerAccess(1, 0);
  if (peer_e != cudaSuccess && peer_e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(peer_e);
  if (peer_e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();

  const size_t N = static_cast<size_t>(mib) * 1024 * 1024 / sizeof(float);
  const size_t bytes = N * sizeof(float);
  float *src0 = nullptr, *compute0 = nullptr, *dst1 = nullptr;
  CUDA_CHECK(cudaMalloc(&src0, bytes));
  CUDA_CHECK(cudaMalloc(&compute0, bytes));
  CUDA_CHECK(cudaMemset(src0, 0x00, bytes));

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMalloc(&dst1, bytes));
  CUDA_CHECK(cudaMemset(dst1, 0xff, bytes));

  // 两条 stream 都建在 GPU0：一条提交 SM compute，一条提交 peer DMA copy。
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t compute_stream, copy_stream;
  CUDA_CHECK(cudaStreamCreate(&compute_stream));
  CUDA_CHECK(cudaStreamCreate(&copy_stream));
  cudaEvent_t compute_start, compute_end, copy_start, copy_end;
  CUDA_CHECK(cudaEventCreate(&compute_start));
  CUDA_CHECK(cudaEventCreate(&compute_end));
  CUDA_CHECK(cudaEventCreate(&copy_start));
  CUDA_CHECK(cudaEventCreate(&copy_end));

  const int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);
  CUDA_CHECK(cudaEventRecord(compute_start, compute_stream));
  busy_compute<<<blocks, threads, 0, compute_stream>>>(compute0, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(compute_end, compute_stream));
  CUDA_CHECK(cudaEventRecord(copy_start, copy_stream));
  CUDA_CHECK(cudaMemcpyPeerAsync(dst1, 1, src0, 0, bytes, copy_stream));
  CUDA_CHECK(cudaEventRecord(copy_end, copy_stream));

  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  CUDA_CHECK(cudaStreamSynchronize(copy_stream));

  float compute_ms = 0.0f, copy_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&compute_ms, compute_start, compute_end));
  CUDA_CHECK(cudaEventElapsedTime(&copy_ms, copy_start, copy_end));
  const float span_ms = std::max(compute_ms, copy_ms);  // 事件零点相同，可取 max 估计并发区间
  const float serial_ms = compute_ms + copy_ms;

  // src0 被置零，所以 peer copy 后 dst1 的抽样值也应为 0。
  std::vector<float> sample(16);
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMemcpy(sample.data(), dst1, sample.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (float v : sample) {
    if (v != 0.0f) {
      std::fprintf(stderr, "FAIL: peer-copy verification mismatch\n");
      return 1;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemcpy(sample.data(), compute0, sample.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (float v : sample) {
    if (!std::isfinite(v)) {
      std::fprintf(stderr, "FAIL: compute verification mismatch\n");
      return 1;
    }
  }

  cudaEventDestroy(compute_start); cudaEventDestroy(compute_end);
  cudaEventDestroy(copy_start); cudaEventDestroy(copy_end);
  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
  cudaFree(src0);
  cudaFree(compute0);
  CUDA_CHECK(cudaSetDevice(1));
  cudaFree(dst1);

  std::printf("PASS: peer copy and SM compute both completed (%d MiB)\n", mib);
  std::printf("compute=%.3f ms, copy=%.3f ms, wall≈%.3f ms (串行则≈%.3f ms)\n",
              compute_ms, copy_ms, span_ms, serial_ms);
  std::printf("wall 明显小于串行说明大概率已并发；是否走 copy engine，仍需 Nsight Systems 确认。\n");
  return 0;
}
