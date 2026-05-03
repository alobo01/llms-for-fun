from __future__ import annotations

import torch


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Applies LayerNorm over the last dimension.

    This wrapper is intentionally explicit even though PyTorch provides a native
    implementation. Keeping a small ops layer now makes later C++ and Rust ports
    easier to mirror conceptually.
    """
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=bias,
        eps=eps,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm implementation for future model families.

    GPT-2 does not use this in phase 0, but adding it now keeps the ops layer
    future-ready for LLaMA-, Gemma-, Mistral-, and Qwen-style upgrades.
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight
