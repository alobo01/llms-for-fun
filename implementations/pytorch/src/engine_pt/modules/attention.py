from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from engine_pt.config.model_config import ModelConfig
from engine_pt.ops.linear import linear
from engine_pt.ops.mask import additive_causal_mask
from engine_pt.ops.softmax import masked_softmax


@dataclass
class AttentionOutput:
    hidden_states: torch.Tensor
    present_k: torch.Tensor | None = None
    present_v: torch.Tensor | None = None


class GPT2Attention:
    """Reference GPT-2 multi-head self-attention.

    Phase 0 goals:
    - correctness and readability first
    - support plain full forward immediately
    - allow optional past KV inputs to ease the later cache milestone
    """

    def __init__(
        self,
        config: ModelConfig,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        out_proj_weight: torch.Tensor,
        out_proj_bias: Optional[torch.Tensor],
    ) -> None:
        if config.attention_type != "mha":
            raise ValueError(f"GPT2Attention expects mha attention, got {config.attention_type}")
        if config.positional_encoding != "learned_absolute":
            raise ValueError(
                "GPT2Attention phase-0 implementation expects learned_absolute positions"
            )

        self.config = config
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.out_proj_weight = out_proj_weight
        self.out_proj_bias = out_proj_bias

        expected_qkv_shape = (3 * config.hidden_size, config.hidden_size)
        if tuple(qkv_weight.shape) != expected_qkv_shape:
            raise ValueError(
                f"qkv_weight must have shape {expected_qkv_shape}, got {tuple(qkv_weight.shape)}"
            )
        if qkv_bias is not None and tuple(qkv_bias.shape) != (3 * config.hidden_size,):
            raise ValueError(
                f"qkv_bias must have shape {(3 * config.hidden_size,)}, got {tuple(qkv_bias.shape)}"
            )
        expected_out_proj_shape = (config.hidden_size, config.hidden_size)
        if tuple(out_proj_weight.shape) != expected_out_proj_shape:
            raise ValueError(
                f"out_proj_weight must have shape {expected_out_proj_shape}, got {tuple(out_proj_weight.shape)}"
            )
        if out_proj_bias is not None and tuple(out_proj_bias.shape) != (config.hidden_size,):
            raise ValueError(
                f"out_proj_bias must have shape {(config.hidden_size,)}, got {tuple(out_proj_bias.shape)}"
            )

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    @property
    def scale(self) -> float:
        return float(self.head_dim) ** -0.5

    def _split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.config.hidden_size
        q, k, v = torch.split(qkv, hidden, dim=-1)
        return q, k, v

    def _reshape_to_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, C], got {tuple(x.shape)}")
        bsz, seq_len, hidden = x.shape
        expected_hidden = num_heads * self.head_dim
        if hidden != expected_hidden:
            raise ValueError(
                f"Hidden size {hidden} does not match num_heads * head_dim {expected_hidden}"
            )
        return x.view(bsz, seq_len, num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape [B, H, T, D], got {tuple(x.shape)}")
        bsz, num_heads, seq_len, head_dim = x.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"Unexpected head shape {(num_heads, head_dim)} for config {(self.num_heads, self.head_dim)}"
            )
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, C], got {tuple(x.shape)}")
        bsz, seq_len, hidden = x.shape
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"Input hidden size {hidden} does not match config.hidden_size {self.config.hidden_size}"
            )
        if (past_k is None) != (past_v is None):
            raise ValueError("past_k and past_v must either both be provided or both be None")

        qkv = linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = self._split_qkv(qkv)

        q = self._reshape_to_heads(q, self.num_heads)
        k = self._reshape_to_heads(k, self.num_kv_heads)
        v = self._reshape_to_heads(v, self.num_kv_heads)

        if past_k is not None and past_v is not None:
            if past_k.ndim != 4 or past_v.ndim != 4:
                raise ValueError("past_k and past_v must have shape [B, kv_heads, S, head_dim]")
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_mask = additive_causal_mask(
            seq_len_q=seq_len,
            seq_len_k=k.shape[-2],
            dtype=attn_scores.dtype,
            device=attn_scores.device,
        )
        attn_probs = masked_softmax(attn_scores, mask=attn_mask, dim=-1)
        context = torch.matmul(attn_probs, v)
        merged = self._merge_heads(context)
        output = linear(merged, self.out_proj_weight, self.out_proj_bias)

        return AttentionOutput(
            hidden_states=output,
            present_k=k if use_cache else None,
            present_v=v if use_cache else None,
        )
