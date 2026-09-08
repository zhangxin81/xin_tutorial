// nvcc -O3 -lineinfo block_slot_demo.cu -o block_slot_demo
// Run: ./block_slot_demo <threads_per_block> <dynamic_smem_bytes> <blocks> <iters>
// Profile with Nsight Compute; useful metrics are listed in README.md.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__global__ void busy_kernel(float* out, int iters) {
  float x = static_cast<float>(blockIdx.x * blockDim.x + threadIdx.x + 1);
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = fmaf(x, 1.0000001f, 0.0000001f);
  }
  if (threadIdx.x == 0) out[blockIdx.x] = x;
}

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 32;
  const int smem = argc > 2 ? std::atoi(argv[2]) : 0;
  const int blocks = argc > 3 ? std::atoi(argv[3]) : 4096;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 200000;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU=%s SMs=%d maxThreadsPerSM=%d maxBlocksPerSM=%d\n",
              prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
              prop.maxBlocksPerMultiProcessor);
  std::printf("threads=%d smem=%d blocks=%d iters=%d\n", threads, smem, blocks, iters);

  float *a=nullptr, *b=nullptr;
  CUDA_CHECK(cudaMalloc(&a, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, blocks * sizeof(float)));
  cudaStream_t s1{}, s2{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

  cudaEvent_t begin{}, end{}, done1{}, done2{};
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventCreate(&done1));
  CUDA_CHECK(cudaEventCreate(&done2));
  CUDA_CHECK(cudaEventRecord(begin));
  CUDA_CHECK(cudaStreamWaitEvent(s1, begin));
  CUDA_CHECK(cudaStreamWaitEvent(s2, begin));

  // Two independent kernels in two streams. Whether they overlap depends on
  // block slots, warp slots, registers, shared memory and execution pipelines.
  busy_kernel<<<blocks, threads, smem, s1>>>(a, iters);
  busy_kernel<<<blocks, threads, smem, s2>>>(b, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(done1, s1));
  CUDA_CHECK(cudaEventRecord(done2, s2));
  CUDA_CHECK(cudaStreamWaitEvent(0, done1));
  CUDA_CHECK(cudaStreamWaitEvent(0, done2));
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  std::printf("two-stream wall time: %.3f ms\n", ms);

  CUDA_CHECK(cudaStreamDestroy(s1));
  CUDA_CHECK(cudaStreamDestroy(s2));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(done1));
  CUDA_CHECK(cudaEventDestroy(done2));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return 0;
}
