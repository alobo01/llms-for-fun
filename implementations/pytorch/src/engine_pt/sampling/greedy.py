from __future__ import annotations

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token ids for the final dimension.

    Args:
        logits: Tensor of shape [..., vocab_size].

    Returns:
        Tensor of shape [...] with integer token ids.
    """
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension")
    return torch.argmax(logits, dim=-1)
