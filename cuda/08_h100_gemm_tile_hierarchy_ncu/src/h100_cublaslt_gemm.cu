#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                                      \
  do {                                                                                        \
    cudaError_t status = (expr);                                                              \
    if (status != cudaSuccess) {                                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                   cudaGetErrorString(status));                                               \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)

#define CHECK_CUBLAS(expr)                                                                    \
  do {                                                                                        \
    cublasStatus_t status = (expr);                                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
      std::fprintf(stderr, "cuBLASLt error at %s:%d: %d\n", __FILE__, __LINE__,              \
                   static_cast<int>(status));                                                 \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)

namespace {

__global__ void fill_inputs(__nv_bfloat16* a, __nv_bfloat16* b, float* c, int64_t total_a,
                            int64_t total_b, int64_t total_c) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < total_a; i += stride) {
    float v = float((i * 13) % 257 - 128) / 128.0f;
    a[i] = __float2bfloat16(v);
  }
  for (int64_t i = tid; i < total_b; i += stride) {
    float v = float((i * 17) % 251 - 125) / 128.0f;
    b[i] = __float2bfloat16(v);
  }
  for (int64_t i = tid; i < total_c; i += stride) {
    c[i] = 0.0f;
  }
}

int get_int_arg(int argc, char** argv, const char* name, int fallback) {
  std::string key = std::string("--") + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(key, 0) == 0) {
      return std::stoi(arg.substr(key.size()));
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const char* name) {
  std::string key = std::string("--") + name;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == key) {
      return true;
    }
  }
  return false;
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " [--m=8192] [--n=8192] [--k=8192] [--iters=200] [--warmup=20] [--verify]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (has_flag(argc, argv, "help")) {
    print_usage(argv[0]);
    return 0;
  }

  int m = get_int_arg(argc, argv, "m", 8192);
  int n = get_int_arg(argc, argv, "n", 8192);
  int k = get_int_arg(argc, argv, "k", 8192);
  int iters = get_int_arg(argc, argv, "iters", 200);
  int warmup = get_int_arg(argc, argv, "warmup", 20);
  bool verify = has_flag(argc, argv, "verify");

  if (m <= 0 || n <= 0 || k <= 0 || iters <= 0 || warmup < 0) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "device=" << prop.name << " sm=" << prop.major << prop.minor << "\n";
  std::cout << "problem=C = A[M,K] x B[K,N], column-major cuBLASLt, bf16 inputs, fp32 output\n";
  std::cout << "M=" << m << " N=" << n << " K=" << k << " warmup=" << warmup
            << " iters=" << iters << "\n";

  int64_t elems_a = int64_t(m) * k;
  int64_t elems_b = int64_t(k) * n;
  int64_t elems_c = int64_t(m) * n;
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, elems_a * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_b, elems_b * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_c, elems_c * sizeof(float)));

  int64_t max_elems = std::max({elems_a, elems_b, elems_c});
  int blocks = static_cast<int>(std::min<int64_t>((max_elems + 255) / 256, 4096));
  fill_inputs<<<blocks, 256>>>(d_a, d_b, d_c, elems_a, elems_b, elems_c);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cublasLtHandle_t lt;
  CHECK_CUBLAS(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t operation_desc = nullptr;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F,
                                        CUDA_R_32F));
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transb, sizeof(transb)));

  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, m, k, m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, n, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, m));

  cublasLtMatmulPreference_t preference = nullptr;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_size = 64ull * 1024 * 1024;
  void* workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));

  constexpr int requested_algorithms = 32;
  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic[requested_algorithms];
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt, operation_desc, a_desc, b_desc, c_desc,
                                              c_desc, preference, requested_algorithms,
                                              heuristic, &returned_results));
  if (returned_results == 0) {
    std::cerr << "No cuBLASLt algorithm found for this shape.\n";
    return EXIT_FAILURE;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    CHECK_CUBLAS(cublasLtMatmul(lt, operation_desc, &alpha, d_a, a_desc, d_b, b_desc,
                                &beta, d_c, c_desc, d_c, c_desc, &heuristic[0].algo,
                                workspace, workspace_size, 0));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUBLAS(cublasLtMatmul(lt, operation_desc, &alpha, d_a, a_desc, d_b, b_desc,
                                &beta, d_c, c_desc, d_c, c_desc, &heuristic[0].algo,
                                workspace, workspace_size, 0));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double avg_ms = double(elapsed_ms) / iters;
  double flops = 2.0 * double(m) * double(n) * double(k);
  double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;
  std::cout << "avg_ms=" << avg_ms << "\n";
  std::cout << "tflops=" << tflops << "\n";

  if (verify) {
    std::vector<float> h_c(16);
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_c) {
      checksum += v;
    }
    std::cout << "checksum_first_16=" << checksum << "\n";
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation_desc));
  CHECK_CUBLAS(cublasLtDestroy(lt));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
