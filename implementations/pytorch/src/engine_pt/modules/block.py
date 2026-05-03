from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from engine_pt.config.model_config import ModelConfig
from engine_pt.modules.attention import GPT2Attention
from engine_pt.modules.mlp import GPT2MLP
from engine_pt.ops.norms import layer_norm


@dataclass
class BlockOutput:
    hidden_states: torch.Tensor
    present_k: torch.Tensor | None = None
    present_v: torch.Tensor | None = None


class GPT2Block:
    """Reference GPT-2 transformer block.

    GPT-2 residual flow:
    - x = x + attn(ln_1(x))
    - x = x + mlp(ln_2(x))
    """

    def __init__(
        self,
        config: ModelConfig,
        ln_1_weight: torch.Tensor,
        ln_1_bias: Optional[torch.Tensor],
        attention: GPT2Attention,
        ln_2_weight: torch.Tensor,
        ln_2_bias: Optional[torch.Tensor],
        mlp: GPT2MLP,
    ) -> None:
        if config.norm_type != "layer_norm":
            raise ValueError(f"GPT2Block expects layer_norm, got {config.norm_type}")

        self.config = config
        self.ln_1_weight = ln_1_weight
        self.ln_1_bias = ln_1_bias
        self.attention = attention
        self.ln_2_weight = ln_2_weight
        self.ln_2_bias = ln_2_bias
        self.mlp = mlp

        expected_norm_shape = (config.hidden_size,)
        for name, tensor in {
            "ln_1_weight": ln_1_weight,
            "ln_2_weight": ln_2_weight,
        }.items():
            if tuple(tensor.shape) != expected_norm_shape:
                raise ValueError(f"{name} must have shape {expected_norm_shape}, got {tuple(tensor.shape)}")
        for name, tensor in {
            "ln_1_bias": ln_1_bias,
            "ln_2_bias": ln_2_bias,
        }.items():
            if tensor is not None and tuple(tensor.shape) != expected_norm_shape:
                raise ValueError(f"{name} must have shape {expected_norm_shape}, got {tuple(tensor.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> BlockOutput:
        ln1 = layer_norm(x, self.ln_1_weight, self.ln_1_bias, eps=self.config.norm_epsilon)
        attn_out = self.attention.forward(ln1, past_k=past_k, past_v=past_v, use_cache=use_cache)
        x = x + attn_out.hidden_states

        ln2 = layer_norm(x, self.ln_2_weight, self.ln_2_bias, eps=self.config.norm_epsilon)
        mlp_out = self.mlp.forward(ln2)
        x = x + mlp_out

        return BlockOutput(
            hidden_states=x,
            present_k=attn_out.present_k,
            present_v=attn_out.present_v,
        )
