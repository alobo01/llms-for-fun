from __future__ import annotations

import torch


def causal_mask(seq_len_q: int, seq_len_k: int, device: torch.device | None = None) -> torch.Tensor:
    """Returns a boolean causal mask of shape [1, 1, seq_len_q, seq_len_k].

    True means the position is visible/allowed.

    This formulation supports both:
    - full-sequence attention where seq_len_q == seq_len_k
    - decode-step attention where seq_len_q may be 1 and seq_len_k includes past cache
    """
    if seq_len_q <= 0 or seq_len_k <= 0:
        raise ValueError("seq_len_q and seq_len_k must be positive")

    q_positions = torch.arange(seq_len_q, device=device).unsqueeze(-1)
    k_positions = torch.arange(seq_len_k, device=device).unsqueeze(0)
    visible = k_positions <= (seq_len_k - seq_len_q + q_positions)
    return visible.unsqueeze(0).unsqueeze(0)


def additive_causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    fill_value: float | None = None,
) -> torch.Tensor:
    """Returns an additive causal mask of shape [1, 1, seq_len_q, seq_len_k].

    Visible entries are 0.0, masked entries are a large negative value.
    """
    visible = causal_mask(seq_len_q=seq_len_q, seq_len_k=seq_len_k, device=device)
    if fill_value is None:
        fill_value = torch.finfo(dtype).min
    zeros = torch.zeros((1, 1, seq_len_q, seq_len_k), dtype=dtype, device=device)
    return torch.where(visible, zeros, torch.full_like(zeros, fill_value))
