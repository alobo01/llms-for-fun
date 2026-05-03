from __future__ import annotations

from dataclasses import dataclass

import torch

from engine_pt.cache.kv_cache import KVCache
from engine_pt.models.gpt2.model import GPT2Model, GPT2ModelOutput


@dataclass
class DecodeStepOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    cache: KVCache


def decode_step(
    model: GPT2Model,
    input_ids: torch.Tensor,
    cache: KVCache,
) -> DecodeStepOutput:
    """Run one autoregressive decode step with an existing cache.

    Args:
        model: GPT-2 model.
        input_ids: New token IDs of shape [B, 1].
        cache: Existing append-only KV cache.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
        raise ValueError(f"input_ids must have shape [B, 1], got {tuple(input_ids.shape)}")

    outputs: GPT2ModelOutput = model.forward(input_ids=input_ids, cache=cache, use_cache=True)

    if outputs.presents is None:
        raise RuntimeError("Model did not return presents while use_cache=True")

    for layer_idx, (present_k, present_v) in enumerate(outputs.presents):
        cache.update(layer_idx, present_k, present_v)

    return DecodeStepOutput(
        logits=outputs.logits,
        hidden_states=outputs.hidden_states,
        cache=cache,
    )
