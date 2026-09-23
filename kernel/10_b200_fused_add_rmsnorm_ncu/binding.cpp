#include <torch/extension.h>

void launch_fused_add_rmsnorm(torch::Tensor hidden_states, torch::Tensor residual,
                              torch::Tensor weight, double eps,
                              torch::Tensor output);
void launch_fused_add_rmsnorm_graph(torch::Tensor hidden_states, torch::Tensor residual,
                                    torch::Tensor weight, double eps,
                                    torch::Tensor output);

void run(torch::Tensor hidden_states, torch::Tensor residual, torch::Tensor weight,
         double eps, torch::Tensor output) {
  launch_fused_add_rmsnorm_graph(hidden_states, residual, weight, eps, output);
}

PYBIND11_MODULE(benchmark_kernel, m) {
  m.def("run", &run);
}
