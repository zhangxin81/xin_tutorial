#include <cuda_runtime.h>

#include <algorithm>
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
  int blocks = 1;
  int threads = 128;
  int elements = 4096;
  int warmup = 5;
  int iters = 50;
  unsigned long long producer_tail_cycles = 800000;
  unsigned long long consumer_prologue_cycles = 700000;
  unsigned long long consumer_body_cycles = 100000;
  std::string mode = "both";
};

__device__ __forceinline__ float burn_cycles(unsigned long long cycles, float x) {
  const unsigned long long start = clock64();
  float v = x;
  while (clock64() - start < cycles) {
    v = fmaf(v, 1.0000001f, 0.0000003f);
#if __CUDA_ARCH__ >= 700
    __nanosleep(32);
#endif
  }
  return v;
}

__global__ void benefit_producer(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 unsigned long long tail_cycles,
                                 int elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (int i = tid; i < elements; i += stride) {
    acc += x[i] * 1.125f;
    y[i] = x[i] + 1.0f;
  }
  __threadfence();

  cudaTriggerProgrammaticLaunchCompletion();

  const float tail = burn_cycles(tail_cycles, acc + threadIdx.x);
  if (tid == 0) {
    y[0] += tail * 0.0f;
  }
}

template <bool UsePdl>
__global__ void benefit_consumer(const float* __restrict__ y,
                                 const float* __restrict__ independent,
                                 float* __restrict__ out,
                                 float* __restrict__ sink,
                                 unsigned long long prologue_cycles,
                                 unsigned long long body_cycles,
                                 int elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  float pre = 0.0f;
  for (int i = tid; i < elements; i += stride) {
    pre += independent[i] * 0.25f;
  }
  pre = burn_cycles(prologue_cycles, pre + threadIdx.x);
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(sink, pre * 1.0e-20f);
  }

  if constexpr (UsePdl) {
    cudaGridDependencySynchronize();
  }

  float dep = 0.0f;
  for (int i = tid; i < elements; i += stride) {
    dep += y[i] * 0.5f;
    out[i] = y[i] + independent[i] + dep * 1.0e-20f;
  }
  const float body = burn_cycles(body_cycles, dep + pre);
  if (tid == 0) {
    sink[0] += body * 1.0e-20f;
  }
}

void launch_pair(bool use_pdl, const Options& opt, cudaStream_t stream,
                 const float* x, const float* independent, float* y, float* out,
                 float* sink) {
  const dim3 grid(opt.blocks, 1, 1);
  const dim3 block(opt.threads, 1, 1);
  benefit_producer<<<grid, block, 0, stream>>>(x, y, opt.producer_tail_cycles,
                                               opt.elements);
  CUDA_CHECK(cudaGetLastError());

  if (!use_pdl) {
    benefit_consumer<false><<<grid, block, 0, stream>>>(
        y, independent, out, sink, opt.consumer_prologue_cycles,
        opt.consumer_body_cycles, opt.elements);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, &benefit_consumer<true>, y, independent,
                                out, sink, opt.consumer_prologue_cycles,
                                opt.consumer_body_cycles, opt.elements));
}

float time_mode(bool use_pdl, const Options& opt, cudaStream_t stream,
                const float* x, const float* independent, float* y, float* out,
                float* sink) {
  for (int i = 0; i < opt.warmup; ++i) {
    launch_pair(use_pdl, opt, stream, x, independent, y, out, sink);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < opt.iters; ++i) {
    launch_pair(use_pdl, opt, stream, x, independent, y, out, sink);
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
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--mode") == 0) opt.mode = need("--mode");
    else if (std::strcmp(argv[i], "--blocks") == 0) opt.blocks = std::atoi(need("--blocks"));
    else if (std::strcmp(argv[i], "--threads") == 0) opt.threads = std::atoi(need("--threads"));
    else if (std::strcmp(argv[i], "--elements") == 0) opt.elements = std::atoi(need("--elements"));
    else if (std::strcmp(argv[i], "--warmup") == 0) opt.warmup = std::atoi(need("--warmup"));
    else if (std::strcmp(argv[i], "--iters") == 0) opt.iters = std::atoi(need("--iters"));
    else if (std::strcmp(argv[i], "--producer-tail-cycles") == 0)
      opt.producer_tail_cycles = std::strtoull(need("--producer-tail-cycles"), nullptr, 10);
    else if (std::strcmp(argv[i], "--consumer-prologue-cycles") == 0)
      opt.consumer_prologue_cycles = std::strtoull(need("--consumer-prologue-cycles"), nullptr, 10);
    else if (std::strcmp(argv[i], "--consumer-body-cycles") == 0)
      opt.consumer_body_cycles = std::strtoull(need("--consumer-body-cycles"), nullptr, 10);
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

  std::vector<float> h_x(opt.elements), h_independent(opt.elements);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  for (float& v : h_x) v = dist(rng);
  for (float& v : h_independent) v = dist(rng);

  float *d_x, *d_independent, *d_y, *d_out_base, *d_out_pdl, *d_sink;
  CUDA_CHECK(cudaMalloc(&d_x, opt.elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_independent, opt.elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, opt.elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_base, opt.elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_pdl, opt.elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), opt.elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_independent, h_independent.data(), opt.elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::printf("GPU=%s cc=%d.%d SMs=%d blocks=%d threads=%d elements=%d producer_tail=%llu consumer_prologue=%llu consumer_body=%llu\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              opt.blocks, opt.threads, opt.elements, opt.producer_tail_cycles,
              opt.consumer_prologue_cycles, opt.consumer_body_cycles);

  if (opt.mode == "baseline" || opt.mode == "both") {
    const float ms = time_mode(false, opt, stream, d_x, d_independent, d_y,
                               d_out_base, d_sink);
    std::printf("baseline_ms=%.4f\n", ms);
  }
  if (opt.mode == "pdl" || opt.mode == "both") {
    const float ms = time_mode(true, opt, stream, d_x, d_independent, d_y,
                               d_out_pdl, d_sink);
    std::printf("pdl_ms=%.4f\n", ms);
  }

  if (opt.mode == "both") {
    std::vector<float> h_base(opt.elements), h_pdl(opt.elements);
    CUDA_CHECK(cudaMemcpy(h_base.data(), d_out_base, opt.elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pdl.data(), d_out_pdl, opt.elements * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < opt.elements; ++i) {
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
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_independent));
  CUDA_CHECK(cudaFree(d_x));
  return 0;
}
