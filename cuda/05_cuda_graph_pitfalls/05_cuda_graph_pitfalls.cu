// 示例五:CUDA Graph capture/replay 的三条硬约束——易错场景与修复。
//
// 对应文章《CUDA Graph:一次录制、多次重放》。capture 把一串 kernel 的调用顺序、
// 参数、地址录成一张图;之后每步一次 cudaGraphLaunch,replay 之间不再有逐 kernel
// 的 CPU launch,提交开销几乎归零(perf 子命令可量化)。代价是三条硬约束:
//
//   1) 形状、地址、参数在录制时固定,replay 不会重算。精确说法:固化的是"参数值
//      和地址",不是"数据"——指针指向的显存内容每次 replay 都是现读的。
//   2) 只有提交到 capture stream 的 GPU 操作会进图;CPU 侧逻辑只在 capture 那一刻
//      执行一次。capture 区域里调 cudaStreamSynchronize / cudaEventSynchronize /
//      同步版 cudaMemcpy 会直接作废 capture(图执行期间也没有 CPU 插手的点)。
//   3) 图执行期间没有 CPU 参与,kernel 背靠背;图只保留你显式表达的依赖边,原来
//      靠 CPU 提交延迟"碰巧"掩盖的 race 会现形。
//
// 每条约束一个 bad demo(复现 bug)+ fix demo(修复姿势),外加启动开销对比:
//   perf         逐 kernel 提交 vs 一次 replay 的 CPU 开销对比(为什么值得用图)
//   s1bad        by-value 参数 + 输出指针被固化:改宿主变量再 replay,输出仍用旧值进旧地址
//   s1fix_data   参数改走 device 内存(图内 H2D 节点从 pinned 读),每步只换数据
//   s1fix_update 用 cudaGraphExecKernelNodeSetParams 每步改参数值和输出地址
//   s2bad        capture 区域里放同步(大声失败)+ CPU 数据准备(安静失败:只执行一次)
//   s2fix        CPU 每步的活在图外干,通过固定地址 pinned 缓冲交给图,同步放 replay 之间
//   s3bad        两条 stream 无显式依赖:靠 CPU 提交间隙碰巧不出事,进图后 race 现形
//   s3fix        显式 event 边把真实依赖写进图:背靠背但有序
//
// 运行(本目录自带独立环境说明,见 README.md;一键脚本 ./build_and_run.sh):
//   ./build_and_run.sh                # 全部 8 个 demo
//   ./build_and_run.sh s1bad          # 单跑某个子命令
// 或手动:
//   nvcc -O2 -std=c++17 05_cuda_graph_pitfalls.cu -o build/cuda_graph_pitfalls
//   ./build/cuda_graph_pitfalls all
//
// profiling:
//   nsys profile -o report05 --force-overwrite true ./build/cuda_graph_pitfalls all
//   nsys-ui report05.nsys-rep    # 看非图段 kernel 之间的 launch gap vs 图段背靠背
//
// 硬件要求:>=1 张 NVIDIA GPU(全部 demo 单 GPU,不需要 P2P/NCCL/NVSHMEM)。

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

namespace {

constexpr int kThreads = 256;
constexpr long long kSpinCycles = 300000;  // ~0.2ms @1.5GHz,只影响 s3 的演示节奏

struct ParamPair {
  float scale;
  float bias;
};

// ---------------------------------------------------------------- kernels

__global__ void axpy_byvalue(const float* in, float* out, float scale, float bias, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * scale + bias;
}

// 参数(scale/bias)从 device 内存现读:地址和形状仍固定,数值每步可换。
__global__ void axpy_devparams(const float* in, float* out, const ParamPair* p, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * p->scale + p->bias;
}

__global__ void plus_one(const float* in, float* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 1.0f;
}

__global__ void tiny_fma(float* buf, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = buf[i];
  for (int r = 0; r < 32; ++r) x = fmaf(x, 1.0000001f, 0.0000001f);
  buf[i] = x;
}

// 自旋 kSpinCycles 个 cycle 后才写:把"写"往后拖,方便观察读者有没有等它。
__global__ void spin_then_write(int* buf, int val, long long spin_cycles) {
  long long start = clock64();
  while (clock64() - start < spin_cycles) {}
  if (threadIdx.x == 0) *buf = val;
}

// 全部线程读 buf[0] 的快照:若与写者并发,快到可以稳稳落在写之前。
__global__ void read_buf0(const int* buf, int* out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = buf[0];
}

// ---------------------------------------------------------------- helpers

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void busy_sleep_ms(double ms) {
  double t0 = now_ms();
  while (now_ms() - t0 < ms) {}
}

cudaGraph_t end_capture(cudaStream_t s) {
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  if (g == nullptr) {
    std::fprintf(stderr, "capture 失败,graph 为空\n");
    std::exit(1);
  }
  return g;
}

cudaGraphExec_t instantiate(cudaGraph_t g) {
  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
  return exec;
}

cudaGraphNode_t first_kernel_node(cudaGraph_t g) {
  size_t num = 0;
  CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &num));
  std::vector<cudaGraphNode_t> nodes(num);
  CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &num));
  for (cudaGraphNode_t nd : nodes) {
    cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
    CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
    if (t == cudaGraphNodeTypeKernel) return nd;
  }
  std::fprintf(stderr, "graph 中没有 kernel 节点\n");
  std::exit(1);
}

std::vector<float> to_host(const float* d, size_t n) {
  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

bool all_close(const std::vector<float>& in, const std::vector<float>& out,
               float scale, float bias) {
  for (size_t i = 0; i < in.size(); ++i) {
    float want = in[i] * scale + bias;
    if (std::fabs(out[i] - want) > 1e-3f) return false;
  }
  return true;
}

bool all_zero(const std::vector<float>& v) {
  for (float x : v) if (x != 0.0f) return false;
  return true;
}

// ---------------------------------------------------------------- perf

// 为什么值得用图:同样的 16 个短 kernel、500 步,逐 kernel 提交 vs 一次 replay。
bool demo_perf() {
  const int K = 16;
  const int STEPS = 500;
  const size_t N = 2048;
  const dim3 grid(16), block(128);

  float* buf = nullptr;
  CUDA_CHECK(cudaMalloc(&buf, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(buf, 0, N * sizeof(float)));
  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  tiny_fma<<<grid, block, 0, s>>>(buf, N);  // 预热,排除首次加载等一次性开销
  CUDA_CHECK(cudaStreamSynchronize(s));

  double t0 = now_ms();
  for (int i = 0; i < STEPS; ++i)
    for (int k = 0; k < K; ++k) tiny_fma<<<grid, block, 0, s>>>(buf, N);
  double t1 = now_ms();
  CUDA_CHECK(cudaStreamSynchronize(s));
  double t2 = now_ms();
  std::printf("非图: 提交 %.2f ms(%.1f us/step, %d 次 launch), 总耗时 %.2f ms\n",
              t1 - t0, (t1 - t0) * 1000.0 / STEPS, STEPS * K, t2 - t0);

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < K; ++k) tiny_fma<<<grid, block, 0, s>>>(buf, N);
  cudaGraph_t g = end_capture(s);
  double t3 = now_ms();
  cudaGraphExec_t exec = instantiate(g);
  double t4 = now_ms();

  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  t0 = now_ms();
  for (int i = 0; i < STEPS; ++i) CUDA_CHECK(cudaGraphLaunch(exec, s));
  t1 = now_ms();
  CUDA_CHECK(cudaStreamSynchronize(s));
  t2 = now_ms();
  std::printf("图:   capture+instantiate 一次性耗时 %.2f ms\n", t4 - t3);
  std::printf("图:   提交 %.2f ms(%.1f us/step, %d 次 launch), 总耗时 %.2f ms\n",
              t1 - t0, (t1 - t0) * 1000.0 / STEPS, STEPS, t2 - t0);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(buf);
  std::printf("结论: 每步 16 次 launch 变 1 次 replay;步频越高、kernel 越短,收益越大。\n");
  return true;
}

// ---------------------------------------------------------------- 场景 1:参数/地址固化

// bad:参数按值传、输出地址写死。replay 前"改了"宿主变量,图无动于衷。
bool demo_s1_bad() {
  std::printf("  capture 时:scale=2, bias=0, 输出到 outB;replay 前改宿主变量\n");
  std::printf("  想:scale=5, bias=1, 输出到 outC。看图听不听。\n");
  const size_t N = 1024;
  std::vector<float> h_in(N);
  for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 7) + 0.5f;

  float *d_in = nullptr, *d_outB = nullptr, *d_outC = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outC, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_outC, 0, N * sizeof(float)));  // 0 作哨兵:真实结果必非 0

  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  float scale = 2.0f, bias = 0.0f;
  float* out = d_outB;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  axpy_byvalue<<<4, kThreads, 0, s>>>(d_in, out, scale, bias, N);
  cudaGraph_t g = end_capture(s);
  cudaGraphExec_t exec = instantiate(g);

  // "新一步":只改宿主变量,直接 replay——参数和地址都在录制时烤死了。
  scale = 5.0f;
  bias = 1.0f;
  out = d_outC;
  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> outB = to_host(d_outB, N);
  std::vector<float> outC = to_host(d_outC, N);
  const bool used_old_params = all_close(h_in, outB, 2.0f, 0.0f);
  const bool ignored_new_out = all_zero(outC);
  std::printf("  outB(out 地址没变)=%s,outC(想写的新地址)=%s\n",
              used_old_params ? "按旧参数 scale=2 重算了一遍" : "内容不符合预期",
              ignored_new_out ? "无人动过(哨兵仍为 0)" : "内容不符合预期");
  std::printf("  -> replay 用的是录制时的参数值和地址,宿主变量的改动无效\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(d_in); cudaFree(d_outB); cudaFree(d_outC);
  return used_old_params && ignored_new_out;
}

// fix A:参数改走 device 内存。图内放一个 H2D 节点,从固定地址的 pinned 缓冲现读;
// 每步只更新 pinned 数据,地址/形状不变,数值随步走。这也是"数据 vs 参数"分界:
// 图固化地址,不固化地址里的内容。
bool demo_s1_fix_data() {
  const size_t N = 1024;
  std::vector<float> h_in(N);
  for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 7) + 0.5f;

  float *d_in = nullptr, *d_out = nullptr;
  ParamPair* d_params = nullptr;
  ParamPair* h_params = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(ParamPair)));
  CUDA_CHECK(cudaMallocHost(&h_params, sizeof(ParamPair)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaMemcpyAsync(d_params, h_params, sizeof(ParamPair),
                             cudaMemcpyHostToDevice, s));
  axpy_devparams<<<4, kThreads, 0, s>>>(d_in, d_out, d_params, N);
  cudaGraph_t g = end_capture(s);
  cudaGraphExec_t exec = instantiate(g);

  bool ok = true;
  const float scales[3] = {3.0f, 4.0f, 5.0f};
  const float biases[3] = {0.0f, 1.0f, 2.0f};
  for (int step = 0; step < 3; ++step) {
    h_params[0] = {scales[step], biases[step]};  // 图外刷新数据
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> out = to_host(d_out, N);
    const bool step_ok = all_close(h_in, out, scales[step], biases[step]);
    std::printf("  step%d: scale=%.0f bias=%.0f -> %s\n",
                step, scales[step], biases[step], step_ok ? "正确" : "错误");
    ok = ok && step_ok;
  }
  std::printf("  -> 参数走 device 内存 + 固定地址 pinned 缓冲,数值每步可换\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_params);
  cudaFreeHost(h_params);
  return ok;
}

// fix B:官方逃生口 cudaGraphExecKernelNodeSetParams,每步直接改节点的参数值
// 和输出地址(gridDim/blockDim 同理可改;更彻底的做法是按形状分桶重新 capture)。
bool demo_s1_fix_update() {
  const size_t N = 1024;
  std::vector<float> h_in(N);
  for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 7) + 0.5f;

  float *d_in = nullptr, *d_outB = nullptr, *d_outC = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outC, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  float scale = 2.0f, bias = 0.0f;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  axpy_byvalue<<<4, kThreads, 0, s>>>(d_in, d_outB, scale, bias, N);
  cudaGraph_t g = end_capture(s);
  cudaGraphExec_t exec = instantiate(g);
  cudaGraphNode_t node = first_kernel_node(g);

  auto update_and_run = [&](float new_scale, float new_bias, float* new_out) {
    cudaKernelNodeParams p{};
    CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &p));
    const float* in = d_in;
    float* out = new_out;
    size_t n = N;
    void* args[5] = {&in, &out, &new_scale, &new_bias, &n};
    p.kernelParams = args;  // SetParams 调用即拷贝,栈变量即可
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &p));
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
  };

  update_and_run(5.0f, 1.0f, d_outB);
  std::vector<float> outB = to_host(d_outB, N);
  const bool step1_ok = all_close(h_in, outB, 5.0f, 1.0f);

  update_and_run(7.0f, 0.5f, d_outC);  // 连输出地址一起换
  std::vector<float> outC = to_host(d_outC, N);
  std::vector<float> outB2 = to_host(d_outB, N);
  const bool step2_ok = all_close(h_in, outC, 7.0f, 0.5f) && all_close(h_in, outB2, 5.0f, 1.0f);

  std::printf("  step1(scale=5,bias=1,out=outB): %s\n", step1_ok ? "正确" : "错误");
  std::printf("  step2(scale=7,bias=0.5,out=outC): %s,outB 未被波及: %s\n",
              all_close(h_in, outC, 7.0f, 0.5f) ? "正确" : "错误",
              all_close(h_in, outB2, 5.0f, 1.0f) ? "是" : "否");
  std::printf("  -> cudaGraphExecKernelNodeSetParams 每步改参数值和地址\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(d_in); cudaFree(d_outB); cudaFree(d_outC);
  return step1_ok && step2_ok;
}

// ---------------------------------------------------------------- 场景 2:CPU 侧的东西不进图

// bad:(a) capture 区域里放同步类调用 -> capture 当场作废(大声失败);
//     (b) 把同步去掉后 capture 成功,但 CPU 的数据准备只在 capture 那刻执行
//         一次 -> replay 每步都在重复拷贝 capture 时刻的旧数据(安静失败)。
bool demo_s2_bad() {
  const size_t N = 1024;
  float *d_in = nullptr, *d_out = nullptr, *d_out2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out2, N * sizeof(float)));
  std::vector<float> h_in(N, 1.0f);
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::printf("  (a) 大声失败:capture 区域里调 cudaStreamSynchronize\n");
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  plus_one<<<4, kThreads, 0, s>>>(d_in, d_out, N);
  cudaError_t sync_err = cudaStreamSynchronize(s);  // 非法:capture 期间不允许同步
  plus_one<<<4, kThreads, 0, s>>>(d_out, d_out2, N);  // capture 已作废
  cudaError_t next_err = cudaGetLastError();
  cudaGraph_t g = nullptr;
  cudaError_t cap_err = cudaStreamEndCapture(s, &g);
  std::printf("      cudaStreamSynchronize -> %s\n", cudaGetErrorString(sync_err));
  std::printf("      作废后的提交       -> %s\n", cudaGetErrorString(next_err));
  std::printf("      cudaStreamEndCapture -> %s, graph=%s\n", cudaGetErrorString(cap_err),
              g == nullptr ? "nullptr" : "非空");
  const bool loud_ok = (sync_err != cudaSuccess) && (g == nullptr);
  if (g != nullptr) cudaGraphDestroy(g);

  std::printf("  (b) 安静失败:CPU 数据准备写进 capture,以为 replay 会重跑\n");
  float *d_x = nullptr, *d_y = nullptr;
  float *pinned_in = nullptr, *pinned_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&pinned_in, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&pinned_out, N * sizeof(float)));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (size_t j = 0; j < N; ++j) pinned_in[j] = 100.0f;  // CPU 逻辑:只在 capture 时执行这一次
  CUDA_CHECK(cudaMemcpyAsync(d_x, pinned_in, N * sizeof(float),
                             cudaMemcpyHostToDevice, s));
  plus_one<<<4, kThreads, 0, s>>>(d_x, d_y, N);
  CUDA_CHECK(cudaMemcpyAsync(pinned_out, d_y, N * sizeof(float),
                             cudaMemcpyDeviceToHost, s));
  cudaGraph_t g2 = end_capture(s);
  cudaGraphExec_t exec = instantiate(g2);

  const float wants[3] = {100.0f, 200.0f, 300.0f};
  int stale = 0;
  for (int step = 0; step < 3; ++step) {
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (std::fabs(pinned_out[0] - (wants[step] + 1.0f)) > 1e-3f) ++stale;
    std::printf("      step%d 想要 %.0f+1, 实得 %.0f\n", step, wants[step], pinned_out[0]);
  }
  const bool silent_ok = (stale == 2);
  std::printf("      -> 填 100 的 CPU 循环只在 capture 时跑了一次;replay 只重放\n");
  std::printf("         图里的 H2D 节点(从固定地址现读),不会重新执行 CPU 代码\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g2);
  cudaStreamDestroy(s);
  cudaFree(d_in); cudaFree(d_out); cudaFree(d_out2); cudaFree(d_x); cudaFree(d_y);
  cudaFreeHost(pinned_in); cudaFreeHost(pinned_out);
  return loud_ok && silent_ok;
}

// fix:CPU 每步的活在图外干;图 = {H2D(pinned->dev), kernel, D2H(dev->pinned)},
// 通过固定地址的 pinned 缓冲进出;同步放在两次 replay 之间。
// 这就是推理框架"静态输入/输出缓冲 + 每步刷缓冲"的标准姿势。
bool demo_s2_fix() {
  const size_t N = 1024;
  float *d_x = nullptr, *d_y = nullptr;
  float *pinned_in = nullptr, *pinned_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&pinned_in, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&pinned_out, N * sizeof(float)));

  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaMemcpyAsync(d_x, pinned_in, N * sizeof(float),
                             cudaMemcpyHostToDevice, s));
  plus_one<<<4, kThreads, 0, s>>>(d_x, d_y, N);
  CUDA_CHECK(cudaMemcpyAsync(pinned_out, d_y, N * sizeof(float),
                             cudaMemcpyDeviceToHost, s));
  cudaGraph_t g = end_capture(s);
  cudaGraphExec_t exec = instantiate(g);

  bool ok = true;
  const float wants[3] = {10.0f, 20.0f, 30.0f};
  for (int step = 0; step < 3; ++step) {
    for (size_t j = 0; j < N; ++j) pinned_in[j] = wants[step];  // 图外刷新
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));  // 同步在 replay 之间
    const bool step_ok = std::fabs(pinned_out[0] - (wants[step] + 1.0f)) <= 1e-3f;
    std::printf("  step%d: 喂 %.0f -> 得 %.0f(%s)\n", step, wants[step],
                pinned_out[0], step_ok ? "正确" : "错误");
    ok = ok && step_ok;
  }
  std::printf("  -> CPU 参与的正确位置:图外准备/检查,固定地址 pinned 做图的 I/O 口\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(d_x); cudaFree(d_y);
  cudaFreeHost(pinned_in); cudaFreeHost(pinned_out);
  return ok;
}

// ---------------------------------------------------------------- 场景 3:依赖边=你显式表达的那几条

// 数一数 out 里有多少元素没读到 42。
int count_stale(const int* d_out, size_t n) {
  std::vector<int> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  int stale = 0;
  for (size_t i = 0; i < n; ++i)
    if (h[i] != 42) ++stale;
  return stale;
}

// bad:写者(W)和读者(R)在两条 stream 上,没有 event 依赖。
// 非图模式下 1ms 的 CPU 提交间隙(模拟框架/Python 开销)碰巧让 W 先跑完;
// capture 进同一张图后两个节点同时起跑,R 稳稳读到旧值——图没有制造 bug,
// 只是撤掉了 CPU 提交延迟这块挡板。
bool demo_s3_bad() {
  const size_t N = static_cast<size_t>(1) << 20;
  int *d_buf = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
  cudaStream_t sA = nullptr, sB = nullptr;
  CUDA_CHECK(cudaStreamCreate(&sA));
  CUDA_CHECK(cudaStreamCreate(&sB));
  cudaEvent_t fork_ev, join_ev;
  CUDA_CHECK(cudaEventCreate(&fork_ev));
  CUDA_CHECK(cudaEventCreate(&join_ev));

  std::printf("  非图 + 1ms CPU 提交间隙(模拟框架开销给的喘息):\n");
  CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(int)));
  CUDA_CHECK(cudaStreamSynchronize(sA));
  spin_then_write<<<1, 32, 0, sA>>>(d_buf, 42, kSpinCycles);
  busy_sleep_ms(1.0);
  read_buf0<<<static_cast<unsigned>(N / kThreads), kThreads, 0, sB>>>(d_buf, d_out, N);
  CUDA_CHECK(cudaStreamSynchronize(sA));
  CUDA_CHECK(cudaStreamSynchronize(sB));
  const int gap_stale = count_stale(d_out, N);
  std::printf("      读到旧值的元素: %d / %zu(%s)\n", gap_stale, N,
              gap_stale == 0 ? "碰巧没事" : "出事了");

  std::printf("  非图、无间隙(去掉挡板的裸跑,本来就有 race):\n");
  CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(int)));
  CUDA_CHECK(cudaStreamSynchronize(sA));
  spin_then_write<<<1, 32, 0, sA>>>(d_buf, 42, kSpinCycles);
  read_buf0<<<static_cast<unsigned>(N / kThreads), kThreads, 0, sB>>>(d_buf, d_out, N);
  CUDA_CHECK(cudaStreamSynchronize(sA));
  CUDA_CHECK(cudaStreamSynchronize(sB));
  std::printf("      读到旧值的元素: %d / %zu\n", count_stale(d_out, N), N);

  std::printf("  capture 成一张图(fork 两个独立根,无显式依赖边),一次 replay:\n");
  CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(int)));
  CUDA_CHECK(cudaStreamSynchronize(sA));
  CUDA_CHECK(cudaStreamBeginCapture(sA, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaEventRecord(fork_ev, sA));  // 记在空 stream 上:图中无边,R 是独立根
  CUDA_CHECK(cudaStreamWaitEvent(sB, fork_ev, 0));
  spin_then_write<<<1, 32, 0, sA>>>(d_buf, 42, kSpinCycles);
  read_buf0<<<static_cast<unsigned>(N / kThreads), kThreads, 0, sB>>>(d_buf, d_out, N);
  CUDA_CHECK(cudaEventRecord(join_ev, sB));
  CUDA_CHECK(cudaStreamWaitEvent(sA, join_ev, 0));
  cudaGraph_t g = end_capture(sA);
  cudaGraphExec_t exec = instantiate(g);
  CUDA_CHECK(cudaGraphLaunch(exec, sA));
  CUDA_CHECK(cudaStreamSynchronize(sA));
  const int graph_stale = count_stale(d_out, N);
  std::printf("      读到旧值的元素: %d / %zu(%s)\n", graph_stale, N,
              graph_stale > 0 ? "race 现形" : "本机未复现,加大 kSpinCycles 重试");
  std::printf("  -> 图只保留 capture 时显式表达的边;CPU launch 延迟一消失,\n");
  std::printf("     原来被提交间隙掩盖的 race 直接暴露\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaEventDestroy(fork_ev);
  cudaEventDestroy(join_ev);
  cudaStreamDestroy(sA);
  cudaStreamDestroy(sB);
  cudaFree(d_buf);
  cudaFree(d_out);
  return gap_stale == 0 && graph_stale > 0;
}

// fix:把真实依赖显式写进图——event 记在 W 之后,capture 成一条真正的边。
// replay 时仍是一次提交、kernel 背靠背,但 R 一定排在 W 后面。
bool demo_s3_fix() {
  const size_t N = static_cast<size_t>(1) << 20;
  int *d_buf = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(int)));
  cudaStream_t sA = nullptr, sB = nullptr;
  CUDA_CHECK(cudaStreamCreate(&sA));
  CUDA_CHECK(cudaStreamCreate(&sB));
  cudaEvent_t dep_ev, join_ev;
  CUDA_CHECK(cudaEventCreate(&dep_ev));
  CUDA_CHECK(cudaEventCreate(&join_ev));

  CUDA_CHECK(cudaStreamBeginCapture(sA, cudaStreamCaptureModeGlobal));
  spin_then_write<<<1, 32, 0, sA>>>(d_buf, 42, kSpinCycles);
  CUDA_CHECK(cudaEventRecord(dep_ev, sA));  // 记在 W 之后 -> 真边 W->R
  CUDA_CHECK(cudaStreamWaitEvent(sB, dep_ev, 0));
  read_buf0<<<static_cast<unsigned>(N / kThreads), kThreads, 0, sB>>>(d_buf, d_out, N);
  CUDA_CHECK(cudaEventRecord(join_ev, sB));
  CUDA_CHECK(cudaStreamWaitEvent(sA, join_ev, 0));
  cudaGraph_t g = end_capture(sA);
  cudaGraphExec_t exec = instantiate(g);

  CUDA_CHECK(cudaGraphLaunch(exec, sA));
  CUDA_CHECK(cudaStreamSynchronize(sA));
  const int stale = count_stale(d_out, N);
  std::printf("  显式 event 边后 replay: 读到旧值的元素 %d / %zu(%s)\n", stale, N,
              stale == 0 ? "正确" : "错误");
  std::printf("  -> 依赖该显式就显式:图执行没有 CPU 喘息点,不能指望碰巧的先后顺序\n");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaEventDestroy(dep_ev);
  cudaEventDestroy(join_ev);
  cudaStreamDestroy(sA);
  cudaStreamDestroy(sB);
  cudaFree(d_buf);
  cudaFree(d_out);
  return stale == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* which = (argc >= 2) ? argv[1] : "all";
  const bool run_all = (std::strcmp(which, "all") == 0);

  struct Named {
    const char* name;
    const char* title;
    bool (*fn)();
  };
  const Named demos[] = {
      {"perf", "为什么快:逐 kernel 提交 vs 一次 replay", demo_perf},
      {"s1bad", "场景1-bad:参数/地址固化,改宿主变量无效", demo_s1_bad},
      {"s1fix_data", "场景1-fixA:参数走 device 内存+固定地址 pinned", demo_s1_fix_data},
      {"s1fix_update", "场景1-fixB:cudaGraphExecKernelNodeSetParams", demo_s1_fix_update},
      {"s2bad", "场景2-bad:capture 里的同步(响)与 CPU 逻辑(静)", demo_s2_bad},
      {"s2fix", "场景2-fix:CPU 在图外,pinned 做图的 I/O 口", demo_s2_fix},
      {"s3bad", "场景3-bad:无显式依赖,进图后 race 现形", demo_s3_bad},
      {"s3fix", "场景3-fix:显式 event 边,背靠背但有序", demo_s3_fix},
  };

  int failures = 0;
  int ran = 0;
  for (const Named& d : demos) {
    if (!run_all && std::strcmp(which, d.name) != 0) continue;
    std::printf("\n===== %s:%s =====\n", d.name, d.title);
    const bool ok = d.fn();
    ++ran;
    if (!ok) ++failures;
    std::printf("[%s] %s\n", ok ? "OK" : "MISMATCH", d.name);
  }
  if (ran == 0) {
    std::fprintf(stderr, "未知子命令: %s\n可用:", which);
    std::fprintf(stderr, " all | perf");
    for (const Named& d : demos)
      if (std::strcmp(d.name, "perf") != 0) std::fprintf(stderr, " | %s", d.name);
    std::fprintf(stderr, "\n");
    return 2;
  }

  std::printf("\n===== 汇总 =====\n");
  std::printf("%s(%d/%d 个 demo 与文档描述一致;bad demo 的成功标准是 bug 被复现)\n",
              failures == 0 ? "全部符合预期" : "存在与描述不符的项", ran - failures, ran);
  return failures == 0 ? 0 : 1;
}
