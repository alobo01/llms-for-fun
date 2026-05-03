from __future__ import annotations

import torch


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits outside the top-k largest values along the last dimension."""
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    vocab_size = logits.shape[-1]
    if top_k >= vocab_size:
        return logits

    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[..., -1:].expand_as(logits)
    masked = logits.masked_fill(logits < threshold, float("-inf"))
    return masked
