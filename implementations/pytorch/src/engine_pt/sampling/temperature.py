from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature.

    temperature=1.0 leaves logits unchanged.
    temperature<1.0 sharpens the distribution.
    temperature>1.0 flattens the distribution.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return logits
    return logits / temperature
