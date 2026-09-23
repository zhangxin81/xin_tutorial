import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = residual + hidden_states
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normalized = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normalized.to(torch.bfloat16)
