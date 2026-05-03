from __future__ import annotations

from typing import Optional

import torch


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies a standard affine transform.

    Contract:
    - x shape: [..., in_features]
    - weight shape: [out_features, in_features]
    - bias shape: [out_features] if provided
    - output shape: [..., out_features]
    """
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got shape {tuple(weight.shape)}")

    in_features = x.shape[-1]
    out_features, weight_in_features = weight.shape
    if in_features != weight_in_features:
        raise ValueError(
            f"Input feature size {in_features} does not match weight in_features {weight_in_features}"
        )

    if bias is not None:
        if bias.ndim != 1:
            raise ValueError(f"bias must be rank-1, got shape {tuple(bias.shape)}")
        if bias.shape[0] != out_features:
            raise ValueError(
                f"bias shape {tuple(bias.shape)} incompatible with out_features {out_features}"
            )

    y = torch.matmul(x, weight.transpose(-1, -2))
    if bias is not None:
        y = y + bias
    return y


class Linear:
    """Small callable wrapper to mirror later cross-language engine structure."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x=x, weight=self.weight, bias=self.bias)
