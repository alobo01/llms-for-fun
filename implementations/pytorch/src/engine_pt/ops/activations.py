from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2 GELU variant, often called gelu_new in Hugging Face configs."""
    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        )
    )


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":
        return gelu(x)
    if activation == "gelu_new":
        return gelu_new(x)
    if activation == "silu":
        return silu(x)
    if activation == "relu":
        return relu(x)
    raise ValueError(f"Unsupported activation: {activation}")
