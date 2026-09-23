#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <unordered_map>

struct alignas(16) Bf16x8 {
  __nv_bfloat162 value[4];
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ Bf16x8 as_bf16x8(uint4 value) {
  Bf16x8 out;
  out.value[0] = *reinterpret_cast<__nv_bfloat162*>(&value.x);
  out.value[1] = *reinterpret_cast<__nv_bfloat162*>(&value.y);
  out.value[2] = *reinterpret_cast<__nv_bfloat162*>(&value.z);
  out.value[3] = *reinterpret_cast<__nv_bfloat162*>(&value.w);
  return out;
}

__device__ __forceinline__ uint4 as_uint4(Bf16x8 value) {
  uint4 out;
  out.x = *reinterpret_cast<unsigned int*>(&value.value[0]);
  out.y = *reinterpret_cast<unsigned int*>(&value.value[1]);
  out.z = *reinterpret_cast<unsigned int*>(&value.value[2]);
  out.w = *reinterpret_cast<unsigned int*>(&value.value[3]);
  return out;
}

__global__ __launch_bounds__(512, 4) void fused_add_rmsnorm(
    const uint4* __restrict__ hidden, const uint4* __restrict__ residual,
    const uint4* __restrict__ weight, uint4* __restrict__ output, float eps) {
  __shared__ float warp_sums[16];
  const int index0 = blockIdx.x * 1024 + threadIdx.x;
  const int index1 = index0 + 512;

  const Bf16x8 h0 = as_bf16x8(hidden[index0]);
  const Bf16x8 r0 = as_bf16x8(residual[index0]);
  const Bf16x8 h1 = as_bf16x8(hidden[index1]);
  const Bf16x8 r1 = as_bf16x8(residual[index1]);

  Bf16x8 sum0;
  Bf16x8 sum1;
  float square_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum0.value[i] = __hadd2(h0.value[i], r0.value[i]);
    const float2 x0 = __bfloat1622float2(sum0.value[i]);
    square_sum = fmaf(x0.x, x0.x, square_sum);
    square_sum = fmaf(x0.y, x0.y, square_sum);

    sum1.value[i] = __hadd2(h1.value[i], r1.value[i]);
    const float2 x1 = __bfloat1622float2(sum1.value[i]);
    square_sum = fmaf(x1.x, x1.x, square_sum);
    square_sum = fmaf(x1.y, x1.y, square_sum);
  }

  square_sum = warp_sum(square_sum);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = square_sum;
  }
  __syncthreads();

  float total = lane < 16 ? warp_sums[lane] : 0.0f;
  total = warp_sum(total);
  total = __shfl_sync(0xffffffff, total, 0);
  const float inv_rms = rsqrtf(total * (1.0f / 8192.0f) + eps);
  const Bf16x8 w0 = as_bf16x8(weight[threadIdx.x]);
  const Bf16x8 w1 = as_bf16x8(weight[threadIdx.x + 512]);
  Bf16x8 out0;
  Bf16x8 out1;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x0 = __bfloat1622float2(sum0.value[i]);
    const __nv_bfloat162 norm0 =
        __float22bfloat162_rn(make_float2(x0.x * inv_rms, x0.y * inv_rms));
    out0.value[i] = __hmul2(norm0, w0.value[i]);

    const float2 x1 = __bfloat1622float2(sum1.value[i]);
    const __nv_bfloat162 norm1 =
        __float22bfloat162_rn(make_float2(x1.x * inv_rms, x1.y * inv_rms));
    out1.value[i] = __hmul2(norm1, w1.value[i]);
  }

  output[index0] = as_uint4(out0);
  output[index1] = as_uint4(out1);
}

__global__ __launch_bounds__(1024, 1) void fused_add_rmsnorm_t1024(
    const uint4* __restrict__ hidden, const uint4* __restrict__ residual,
    const uint4* __restrict__ weight, uint4* __restrict__ output, float eps) {
  __shared__ float warp_sums[32];
  const int index = blockIdx.x * 1024 + threadIdx.x;

  const Bf16x8 h = as_bf16x8(hidden[index]);
  const Bf16x8 r = as_bf16x8(residual[index]);
  Bf16x8 sum;
  float square_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum.value[i] = __hadd2(h.value[i], r.value[i]);
    const float2 x = __bfloat1622float2(sum.value[i]);
    square_sum = fmaf(x.x, x.x, square_sum);
    square_sum = fmaf(x.y, x.y, square_sum);
  }

  square_sum = warp_sum(square_sum);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = square_sum;
  }
  __syncthreads();

  float total = warp_sums[lane];
  total = warp_sum(total);
  total = __shfl_sync(0xffffffff, total, 0);
  const float inv_rms = rsqrtf(total * (1.0f / 8192.0f) + eps);
  const Bf16x8 w = as_bf16x8(weight[threadIdx.x]);
  Bf16x8 out;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __bfloat1622float2(sum.value[i]);
    const __nv_bfloat162 norm =
        __float22bfloat162_rn(make_float2(x.x * inv_rms, x.y * inv_rms));
    out.value[i] = __hmul2(norm, w.value[i]);
  }

  output[index] = as_uint4(out);
}

void launch_fused_add_rmsnorm(torch::Tensor hidden_states, torch::Tensor residual,
                              torch::Tensor weight, double eps,
                              torch::Tensor output) {
  const int rows = hidden_states.numel() / 8192;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (rows == 256 || rows == 1024) {
    fused_add_rmsnorm_t1024<<<rows, 1024, 0, stream>>>(
        reinterpret_cast<const uint4*>(hidden_states.data_ptr()),
        reinterpret_cast<const uint4*>(residual.data_ptr()),
        reinterpret_cast<const uint4*>(weight.data_ptr()),
        reinterpret_cast<uint4*>(output.data_ptr()), static_cast<float>(eps));
    return;
  }
  fused_add_rmsnorm<<<rows, 512, 0, stream>>>(
      reinterpret_cast<const uint4*>(hidden_states.data_ptr()),
      reinterpret_cast<const uint4*>(residual.data_ptr()),
      reinterpret_cast<const uint4*>(weight.data_ptr()),
      reinterpret_cast<uint4*>(output.data_ptr()), static_cast<float>(eps));
}

struct GraphCacheEntry {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t kernel_node = nullptr;
  bool t1024 = false;
  int rows = 0;
};

GraphCacheEntry& get_graph_cache(int rows) {
  static std::unordered_map<int, GraphCacheEntry> cache;
  return cache[rows];
}

void launch_fused_add_rmsnorm_graph(torch::Tensor hidden_states, torch::Tensor residual,
                                    torch::Tensor weight, double eps,
                                    torch::Tensor output) {
  const int rows = hidden_states.numel() / 8192;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_t1024 = rows == 256 || rows == 1024;
  GraphCacheEntry& cache = get_graph_cache(rows);

  const uint4* hidden_ptr = reinterpret_cast<const uint4*>(hidden_states.data_ptr());
  const uint4* residual_ptr = reinterpret_cast<const uint4*>(residual.data_ptr());
  const uint4* weight_ptr = reinterpret_cast<const uint4*>(weight.data_ptr());
  uint4* output_ptr = reinterpret_cast<uint4*>(output.data_ptr());
  float eps_f = static_cast<float>(eps);
  void* kernel_args[] = {&hidden_ptr, &residual_ptr, &weight_ptr, &output_ptr, &eps_f};

  dim3 grid(rows, 1, 1);
  dim3 block(use_t1024 ? 1024 : 512, 1, 1);
  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(use_t1024 ? fused_add_rmsnorm_t1024 : fused_add_rmsnorm);
  params.gridDim = grid;
  params.blockDim = block;
  params.sharedMemBytes = 0;
  params.kernelParams = kernel_args;
  params.extra = nullptr;

  if (cache.exec == nullptr || cache.rows != rows || cache.t1024 != use_t1024) {
    if (cache.exec != nullptr) {
      cudaGraphExecDestroy(cache.exec);
      cudaGraphDestroy(cache.graph);
    }
    cudaGraphCreate(&cache.graph, 0);
    cudaGraphAddKernelNode(&cache.kernel_node, cache.graph, nullptr, 0, &params);
    cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
    cache.rows = rows;
    cache.t1024 = use_t1024;
  } else {
    cudaGraphExecKernelNodeSetParams(cache.exec, cache.kernel_node, &params);
  }

  cudaGraphLaunch(cache.exec, stream);
}
