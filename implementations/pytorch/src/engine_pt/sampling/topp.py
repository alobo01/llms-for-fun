from __future__ import annotations

import torch


def apply_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """Nucleus filtering over the last dimension.

    Keeps the smallest prefix of tokens whose cumulative probability >= top_p.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if min_tokens_to_keep < 1:
        raise ValueError("min_tokens_to_keep must be >= 1")

    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove_mask = cumulative_probs > top_p
    # Keep at least min_tokens_to_keep tokens.
    sorted_remove_mask[..., :min_tokens_to_keep] = False
    # Shift the mask right so the first token that crosses top_p is kept.
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove_mask, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))
