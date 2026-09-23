from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
_LOCAL_BINDING = ROOT / "_binding_local.cpp"
_BINDING_TEXT = (ROOT / "binding.cpp").read_text().replace(
    "PYBIND11_MODULE(benchmark_kernel, m)", "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)"
)
if not _LOCAL_BINDING.exists() or _LOCAL_BINDING.read_text() != _BINDING_TEXT:
    _LOCAL_BINDING.write_text(_BINDING_TEXT)

_ext = load(
    name="kernel69_cuda_sm100_graph_replay_local",
    sources=[str(_LOCAL_BINDING), str(ROOT / "kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


@torch.no_grad()
def run(hidden_states, residual, weight, eps, output=None):
    if output is None:
        output = torch.empty_like(hidden_states)
    _ext.run(hidden_states, residual, weight, eps, output)
    return output
