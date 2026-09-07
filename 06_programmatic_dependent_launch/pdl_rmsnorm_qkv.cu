#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err__));                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct Options {
  int m = 64;
  int k = 4096;
  int n = 768;
  int warmup = 5;
  int iters = 20;
  unsigned long long tail_cycles = 300000;
  std::string mode = "both";
};

__global__ void rmsnorm_producer(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 float* __restrict__ y, int m, int k, float eps,
                                 unsigned long long tail_cycles) {
  extern __shared__ float scratch[];
  const int row = blockIdx.x;
  if (row >= m) return;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    const float v = x[row * k + col];
    sum_sq += v * v;
  }
  scratch[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scratch[0] = rsqrtf(scratch[0] / static_cast<float>(k) + eps);
  }
  __syncthreads();
  const float inv_rms = scratch[0];

  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    y[row * k + col] = x[row * k + col] * inv_rms * gamma[col];
  }
  __threadfence();

  // The dependent kernel may only consume data produced before this trigger.
  // The synthetic tail below is independent epilogue work used to make overlap visible.
  cudaTriggerProgrammaticLaunchCompletion();

  const unsigned long long start = clock64();
  while (clock64() - start < tail_cycles) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
  }
}

template <bool UsePdl>
__global__ void qkv_consumer(const float* __restrict__ y,
                             const float* __restrict__ weight,
                             float* __restrict__ out,
                             float* __restrict__ prefetch_sink, int m, int k,
                             int n) {
  float touched = 0.0f;
  const int linear_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int linear_stride = gridDim.x * blockDim.x;
  for (int idx = linear_tid; idx < k * n; idx += linear_stride) {
    touched += weight[idx];
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(prefetch_sink, touched * 1.0e-20f);
  }

  if constexpr (UsePdl) {
    cudaGridDependencySynchronize();
  }

  for (int idx = linear_tid; idx < m * n; idx += linear_stride) {
    const int row = idx / n;
    const int col = idx - row * n;
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
      acc = fmaf(y[row * k + inner], weight[inner * n + col], acc);
    }
    out[idx] = acc;
  }
}

void launch_pair(bool use_pdl, const Options& opt, int consumer_grid,
                 cudaStream_t stream, const float* x, const float* gamma,
                 float* y, const float* weight, float* out, float* sink) {
  constexpr int rms_threads = 256;
  constexpr int gemm_threads = 128;
  const size_t rms_smem = rms_threads * sizeof(float);

  rmsnorm_producer<<<opt.m, rms_threads, rms_smem, stream>>>(
      x, gamma, y, opt.m, opt.k, 1.0e-6f, opt.tail_cycles);
  CUDA_CHECK(cudaGetLastError());

  if (!use_pdl) {
    qkv_consumer<false><<<consumer_grid, gemm_threads, 0, stream>>>(
        y, weight, out, sink, opt.m, opt.k, opt.n);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(consumer_grid, 1, 1);
  config.blockDim = dim3(gemm_threads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, &qkv_consumer<true>, y, weight, out,
                                sink, opt.m, opt.k, opt.n));
}

float time_mode(bool use_pdl, const Options& opt, int consumer_grid,
                cudaStream_t stream, const float* x, const float* gamma,
                float* y, const float* weight, float* out, float* sink) {
  for (int i = 0; i < opt.warmup; ++i) {
    launch_pair(use_pdl, opt, consumer_grid, stream, x, gamma, y, weight, out,
                sink);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < opt.iters; ++i) {
    launch_pair(use_pdl, opt, consumer_grid, stream, x, gamma, y, weight, out,
                sink);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(opt.iters);
}

Options parse_options(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need_value = [&](const char* flag) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--mode") == 0) opt.mode = need_value("--mode");
    else if (std::strcmp(argv[i], "--m") == 0) opt.m = std::atoi(need_value("--m"));
    else if (std::strcmp(argv[i], "--k") == 0) opt.k = std::atoi(need_value("--k"));
    else if (std::strcmp(argv[i], "--n") == 0) opt.n = std::atoi(need_value("--n"));
    else if (std::strcmp(argv[i], "--warmup") == 0) opt.warmup = std::atoi(need_value("--warmup"));
    else if (std::strcmp(argv[i], "--iters") == 0) opt.iters = std::atoi(need_value("--iters"));
    else if (std::strcmp(argv[i], "--tail-cycles") == 0)
      opt.tail_cycles = std::strtoull(need_value("--tail-cycles"), nullptr, 10);
    else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (opt.mode != "baseline" && opt.mode != "pdl" && opt.mode != "both") {
    std::fprintf(stderr, "--mode must be baseline, pdl, or both\n");
    std::exit(EXIT_FAILURE);
  }
  return opt;
}

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 9) {
    std::fprintf(stderr, "PDL requires compute capability 9.0+. Found %d.%d\n",
                 prop.major, prop.minor);
    return EXIT_FAILURE;
  }
  const int consumer_grid = prop.multiProcessorCount;

  const size_t x_count = static_cast<size_t>(opt.m) * opt.k;
  const size_t w_count = static_cast<size_t>(opt.k) * opt.n;
  const size_t out_count = static_cast<size_t>(opt.m) * opt.n;

  std::vector<float> h_x(x_count), h_gamma(opt.k), h_weight(w_count);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  for (float& v : h_x) v = dist(rng);
  for (float& v : h_gamma) v = 1.0f + dist(rng);
  for (float& v : h_weight) v = dist(rng);

  float *d_x, *d_gamma, *d_y, *d_weight, *d_out_base, *d_out_pdl, *d_sink;
  CUDA_CHECK(cudaMalloc(&d_x, x_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, opt.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, x_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, w_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_base, out_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_pdl, out_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), opt.k * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), w_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::printf("GPU=%s cc=%d.%d SMs=%d shape=[%d,%d]x[%d,%d] tail_cycles=%llu\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              opt.m, opt.k, opt.k, opt.n, opt.tail_cycles);

  if (opt.mode == "baseline" || opt.mode == "both") {
    const float ms = time_mode(false, opt, consumer_grid, stream, d_x, d_gamma,
                               d_y, d_weight, d_out_base, d_sink);
    std::printf("baseline_ms=%.4f\n", ms);
  }
  if (opt.mode == "pdl" || opt.mode == "both") {
    const float ms = time_mode(true, opt, consumer_grid, stream, d_x, d_gamma,
                               d_y, d_weight, d_out_pdl, d_sink);
    std::printf("pdl_ms=%.4f\n", ms);
  }

  if (opt.mode == "both") {
    std::vector<float> h_base(out_count), h_pdl(out_count);
    CUDA_CHECK(cudaMemcpy(h_base.data(), d_out_base, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pdl.data(), d_out_pdl, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < out_count; ++i) {
      const float abs_err = std::abs(h_base[i] - h_pdl[i]);
      const float denom = std::max(1.0e-6f, std::abs(h_base[i]));
      max_abs = std::max(max_abs, abs_err);
      max_rel = std::max(max_rel, abs_err / denom);
    }
    std::printf("max_abs_error=%.8g max_rel_error=%.8g %s\n", max_abs, max_rel,
                (max_abs == 0.0f ? "PASS" : "CHECK"));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out_pdl));
  CUDA_CHECK(cudaFree(d_out_base));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_x));
  return 0;
}
