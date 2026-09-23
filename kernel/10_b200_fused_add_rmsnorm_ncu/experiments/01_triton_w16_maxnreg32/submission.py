import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm(hidden, residual, weight, output, eps: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 8192)
    offset = row * 8192 + cols

    # PyTorch rounds the residual sum to BF16 before the FP32 reduction.
    x = (tl.load(hidden + offset).to(tl.float32) + tl.load(residual + offset).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / 8192.0) + eps)
    normalized = (x * inv_rms).to(tl.bfloat16)
    out = (normalized.to(tl.float32) * tl.load(weight + cols).to(tl.float32)).to(tl.bfloat16)
    tl.store(output + offset, out)


@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // 8192
    _fused_add_rmsnorm[(rows,)](hidden_states, residual, weight, output, eps=eps, num_warps=16, maxnreg=32)
    return output
