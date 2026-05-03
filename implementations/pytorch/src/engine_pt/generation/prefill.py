from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from engine_pt.cache.kv_cache import KVCache
from engine_pt.models.gpt2.model import GPT2Model, GPT2ModelOutput


@dataclass
class PrefillOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    cache: KVCache


def prefill(
    model: GPT2Model,
    input_ids: torch.Tensor,
    cache: Optional[KVCache] = None,
) -> PrefillOutput:
    """Run prompt prefill and return populated KV cache.

    Args:
        model: GPT-2 model.
        input_ids: Prompt token IDs of shape [B, T].
        cache: Optional existing cache. If omitted, a new empty cache is created.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")

    working_cache = cache if cache is not None else KVCache.empty_from_config(model.config)
    outputs: GPT2ModelOutput = model.forward(input_ids=input_ids, cache=working_cache, use_cache=True)

    if outputs.presents is None:
        raise RuntimeError("Model did not return presents while use_cache=True")

    for layer_idx, (present_k, present_v) in enumerate(outputs.presents):
        working_cache.update(layer_idx, present_k, present_v)

    return PrefillOutput(
        logits=outputs.logits,
        hidden_states=outputs.hidden_states,
        cache=working_cache,
    )
