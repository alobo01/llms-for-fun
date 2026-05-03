from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from engine_pt.generation.decode import decode_step
from engine_pt.generation.prefill import prefill
from engine_pt.sampling.greedy import greedy_sample
from engine_pt.sampling.temperature import apply_temperature
from engine_pt.sampling.topk import apply_top_k
from engine_pt.sampling.topp import apply_top_p


@dataclass
class SamplingConfig:
    mode: str = "greedy"  # greedy | sample
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class GenerationOutput:
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    full_sequence_ids: torch.Tensor
    steps: int
    stopped_on_eos: bool



def sample_next_token(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    """Sample next token ids from logits of shape [batch, vocab_size]."""
    if logits.ndim != 2:
        raise ValueError(f"expected logits shape [batch, vocab_size], got {tuple(logits.shape)}")

    if sampling.mode == "greedy":
        return greedy_sample(logits)

    filtered = apply_temperature(logits, sampling.temperature)
    if sampling.top_k is not None:
        filtered = apply_top_k(filtered, sampling.top_k)
    if sampling.top_p is not None:
        filtered = apply_top_p(filtered, sampling.top_p)

    probs = torch.softmax(filtered, dim=-1)
    if sampling.seed is not None:
        generator = torch.Generator(device=probs.device)
        generator.manual_seed(sampling.seed)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)



def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    sampling: Optional[SamplingConfig] = None,
) -> GenerationOutput:
    """Autoregressive generation using prefill + one-token decode."""
    if input_ids.ndim != 2:
        raise ValueError(f"expected input_ids shape [batch, seq], got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    sampling = sampling or SamplingConfig()
    prefill_out = prefill(model, input_ids)

    all_ids = input_ids.clone()
    generated = []
    cache = prefill_out.cache
    stopped_on_eos = False

    last_logits = prefill_out.logits[:, -1, :]
    next_token = sample_next_token(last_logits, sampling)

    for step in range(max_new_tokens):
        generated.append(next_token.unsqueeze(1))
        all_ids = torch.cat([all_ids, next_token.unsqueeze(1)], dim=1)

        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            stopped_on_eos = True
            break

        decode_out = decode_step(model, next_token.unsqueeze(1), cache)
        cache = decode_out.cache
        next_token = sample_next_token(decode_out.logits[:, -1, :], sampling)

    generated_ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    )

    return GenerationOutput(
        input_ids=input_ids,
        generated_ids=generated_ids,
        full_sequence_ids=all_ids,
        steps=generated_ids.shape[1],
        stopped_on_eos=stopped_on_eos,
    )
