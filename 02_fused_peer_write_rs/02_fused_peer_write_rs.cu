// 示例二：GEMM + ReduceScatter 的 peer write 数据流。
//
// 对应文章《GPU 计算与通信融合入门》第七章。属于“prologue/epilogue 融合”路线：
// 计算 CTA 在 GEMM epilogue 中直接把 partial result 写进 output owner GPU 的
// rank slot，粒度是 CTA / tile。
//
// 数据流：
//   - 两张 GPU 分别保存 K 维的一半（A、B 的 K/2 分片），各自计算 partial C。
//   - kernel 内每个 thread 根据 output row 判断 owner（GPU0 还是 GPU1），
//     直接把 partial result peer write 到 owner GPU 上自己的 rank slot。
//   - 两个 rank 写入不同 slot，因此不要求跨 GPU atomic；owner 随后执行一次
//     本地 reduce，得到完整的 ReduceScatter 输出分片。
//
// 说明：这里 GEMM 是朴素 O(MNK) 教学实现。生产版本会使用 Tensor Core tile、
// 双缓冲、vectorized remote write 和更细的 signal。
//
// 运行（本目录自带独立环境说明，见 README.md；一键脚本 ./build_and_run.sh）：
//   ./build_and_run.sh                     # 默认 M=N=K=512
//   ./build_and_run.sh 1024 1024 1024      # 自定义规模（M、K 需为偶数）
// 或手动：
//   nvcc -O2 -std=c++17 02_fused_peer_write_rs.cu -o build/fused_peer_write_rs
//   ./build/fused_peer_write_rs
//
// profiling：
//   nsys profile -o 02_peer_write --force-overwrite true ./build/fused_peer_write_rs
//   nsys-ui 02_peer_write.nsys-rep
//
// 硬件要求：同一台机器上至少 2 张支持 P2P 的 NVIDIA GPU。

#include <cuda_runtime.h>
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

// 每个 rank 保存 A、B 的 K/2 分片，计算 partial C。
// epilogue 不写本地完整 C，而是直接写到 output owner GPU 的对应 slot。
__global__ void partial_gemm_peer_write(
    const float* a_k_shard, const float* b_k_shard,
    float* owner0_slots, float* owner1_slots,
    int rank, int m, int n, int k_local) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int k = 0; k < k_local; ++k) {
    acc += a_k_shard[row * k_local + k] * b_k_shard[k * n + col];
  }

  const int rows_per_owner = m / 2;
  const int owner = row / rows_per_owner;
  const int local_row = row % rows_per_owner;
  const size_t part_elems = static_cast<size_t>(rows_per_owner) * n;
  float* target = owner == 0 ? owner0_slots : owner1_slots;

  // rank 0/1 写不同 slot，因此无需 remote atomic。
  target[static_cast<size_t>(rank) * part_elems + local_row * n + col] = acc;
}

__global__ void reduce_rank_slots(const float* slots, float* out, size_t elems) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elems) out[i] = slots[i] + slots[elems + i];
}

static void enable_peer_or_die(int from, int to) {
  int can = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can, from, to));
  if (!can) {
    std::fprintf(stderr, "GPU %d cannot directly access GPU %d\n", from, to);
    std::exit(2);
  }
  CUDA_CHECK(cudaSetDevice(from));
  cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
  if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
}

// 用法: ./fused_peer_write_rs [M] [N] [K]，默认 512 512 512。
static void parse_args(int argc, char** argv, int* m, int* n, int* k) {
  *m = 512; *n = 512; *k = 512;
  if (argc >= 2) *m = std::atoi(argv[1]);
  if (argc >= 3) *n = std::atoi(argv[2]);
  if (argc >= 4) *k = std::atoi(argv[3]);
  if (*m <= 0 || *n <= 0 || *k <= 0 || (*m % 2) != 0 || (*k % 2) != 0) {
    std::fprintf(stderr, "要求 M、N、K 为正数，且 M、K 为偶数（2 卡各分一半）\n");
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  int M = 0, N = 0, K = 0;
  parse_args(argc, argv, &M, &N, &K);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::fprintf(stderr, "需要至少两张 GPU\n");
    return 2;
  }
  enable_peer_or_die(0, 1);
  enable_peer_or_die(1, 0);

  constexpr int WORLD = 2;
  const int K_LOCAL = K / WORLD;
  const int ROWS_LOCAL = M / WORLD;
  const size_t part_elems = static_cast<size_t>(ROWS_LOCAL) * N;

  float *a[WORLD], *b[WORLD];
  float *owner_slots[WORLD], *owner_out[WORLD];
  cudaStream_t stream[WORLD];
  cudaEvent_t gemm_start[WORLD], gemm_end[WORLD];

  // A、B 全 1，则 C = A·B 的每个元素都等于 K，校验无需 CPU 参考实现。
  std::vector<float> h_a(static_cast<size_t>(M) * K_LOCAL, 1.0f);
  std::vector<float> h_b(static_cast<size_t>(K_LOCAL) * N, 1.0f);

  for (int rank = 0; rank < WORLD; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream[rank]));
    CUDA_CHECK(cudaEventCreate(&gemm_start[rank]));
    CUDA_CHECK(cudaEventCreate(&gemm_end[rank]));
    CUDA_CHECK(cudaMalloc(&a[rank], h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b[rank], h_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&owner_slots[rank], WORLD * part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&owner_out[rank], part_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(a[rank], h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice, stream[rank]));
    CUDA_CHECK(cudaMemcpyAsync(b[rank], h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice, stream[rank]));
    CUDA_CHECK(cudaMemsetAsync(owner_slots[rank], 0, WORLD * part_elems * sizeof(float), stream[rank]));
  }

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  for (int rank = 0; rank < WORLD; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaEventRecord(gemm_start[rank], stream[rank]));
    partial_gemm_peer_write<<<grid, block, 0, stream[rank]>>>(
        a[rank], b[rank], owner_slots[0], owner_slots[1], rank, M, N, K_LOCAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(gemm_end[rank], stream[rank]));
  }

  // 先确保两个 rank 的本地/远端 epilogue 写全部结束。
  for (int rank = 0; rank < WORLD; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(stream[rank]));
  }

  cudaEvent_t reduce_start[WORLD], reduce_end[WORLD];
  for (int owner = 0; owner < WORLD; ++owner) {
    CUDA_CHECK(cudaSetDevice(owner));
    CUDA_CHECK(cudaEventCreate(&reduce_start[owner]));
    CUDA_CHECK(cudaEventCreate(&reduce_end[owner]));
    int threads = 256;
    int blocks = static_cast<int>((part_elems + threads - 1) / threads);
    CUDA_CHECK(cudaEventRecord(reduce_start[owner], stream[owner]));
    reduce_rank_slots<<<blocks, threads, 0, stream[owner]>>>(
        owner_slots[owner], owner_out[owner], part_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(reduce_end[owner], stream[owner]));
  }

  bool ok = true;
  for (int owner = 0; owner < WORLD; ++owner) {
    std::vector<float> host(part_elems);
    CUDA_CHECK(cudaSetDevice(owner));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), owner_out[owner], part_elems * sizeof(float), cudaMemcpyDeviceToHost, stream[owner]));
    CUDA_CHECK(cudaStreamSynchronize(stream[owner]));
    for (float v : host) {
      if (std::fabs(v - static_cast<float>(K)) > 1e-5f) { ok = false; break; }
    }
  }

  float gemm_ms[WORLD], reduce_ms[WORLD];
  for (int rank = 0; rank < WORLD; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaEventElapsedTime(&gemm_ms[rank], gemm_start[rank], gemm_end[rank]));
    CUDA_CHECK(cudaEventElapsedTime(&reduce_ms[rank], reduce_start[rank], reduce_end[rank]));
    cudaEventDestroy(gemm_start[rank]);
    cudaEventDestroy(gemm_end[rank]);
    cudaEventDestroy(reduce_start[rank]);
    cudaEventDestroy(reduce_end[rank]);
  }

  for (int rank = 0; rank < WORLD; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    cudaFree(a[rank]); cudaFree(b[rank]);
    cudaFree(owner_slots[rank]); cudaFree(owner_out[rank]);
    cudaStreamDestroy(stream[rank]);
  }

  if (!ok) {
    std::fprintf(stderr, "FAIL: output mismatch\n");
    return 1;
  }
  std::printf("PASS: every ReduceScatter output element equals K=%d\n", K);
  std::printf("M=%d N=%d K=%d | partial GEMM: %.3f / %.3f ms, local reduce: %.3f / %.3f ms\n",
              M, N, K, gemm_ms[0], gemm_ms[1], reduce_ms[0], reduce_ms[1]);
  std::printf("remote epilogue write 是否占用额外时间，请用 Nsight Systems 对照观察。\n");
  return 0;
}
