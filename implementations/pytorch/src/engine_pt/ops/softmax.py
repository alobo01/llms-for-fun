from __future__ import annotations

from typing import Optional

import torch


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Uses max-subtraction explicitly so the behavior is clear in the engine layer,
    even though torch.softmax is already numerically stable internally.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    shifted = x - max_vals
    exp_shifted = torch.exp(shifted)
    denom = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / denom


def masked_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    mask_fill_value: Optional[float] = None,
) -> torch.Tensor:
    """Applies an additive or boolean mask before stable softmax.

    Supported mask forms:
    - boolean mask: False entries are masked out
    - additive mask: values are added directly to scores before softmax

    Broadcasting is delegated to PyTorch.
    """
    if mask is None:
        return stable_softmax(scores, dim=dim)

    if mask.dtype == torch.bool:
        fill_value = mask_fill_value
        if fill_value is None:
            fill_value = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~mask, fill_value)
    else:
        masked_scores = scores + mask

    return stable_softmax(masked_scores, dim=dim)
