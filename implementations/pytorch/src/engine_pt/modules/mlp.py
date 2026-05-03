from __future__ import annotations

from typing import Optional

import torch

from engine_pt.config.model_config import ModelConfig
from engine_pt.ops.activations import apply_activation
from engine_pt.ops.linear import linear


class GPT2MLP:
    """Reference GPT-2 feed-forward network.

    Phase 0 supports the GPT-2 GELU FFN path only.
    """

    def __init__(
        self,
        config: ModelConfig,
        fc_in_weight: torch.Tensor,
        fc_in_bias: Optional[torch.Tensor],
        fc_out_weight: torch.Tensor,
        fc_out_bias: Optional[torch.Tensor],
    ) -> None:
        if config.mlp_type != "gelu_ffn":
            raise ValueError(f"GPT2MLP expects gelu_ffn, got {config.mlp_type}")
        if config.activation != "gelu_new":
            raise ValueError(f"GPT2MLP expects gelu_new activation, got {config.activation}")

        self.config = config
        self.fc_in_weight = fc_in_weight
        self.fc_in_bias = fc_in_bias
        self.fc_out_weight = fc_out_weight
        self.fc_out_bias = fc_out_bias

        expected_fc_in = (config.intermediate_size, config.hidden_size)
        expected_fc_out = (config.hidden_size, config.intermediate_size)
        if tuple(fc_in_weight.shape) != expected_fc_in:
            raise ValueError(
                f"fc_in_weight must have shape {expected_fc_in}, got {tuple(fc_in_weight.shape)}"
            )
        if fc_in_bias is not None and tuple(fc_in_bias.shape) != (config.intermediate_size,):
            raise ValueError(
                f"fc_in_bias must have shape {(config.intermediate_size,)}, got {tuple(fc_in_bias.shape)}"
            )
        if tuple(fc_out_weight.shape) != expected_fc_out:
            raise ValueError(
                f"fc_out_weight must have shape {expected_fc_out}, got {tuple(fc_out_weight.shape)}"
            )
        if fc_out_bias is not None and tuple(fc_out_bias.shape) != (config.hidden_size,):
            raise ValueError(
                f"fc_out_bias must have shape {(config.hidden_size,)}, got {tuple(fc_out_bias.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, C], got {tuple(x.shape)}")
        h = linear(x, self.fc_in_weight, self.fc_in_bias)
        h = apply_activation(h, self.config.activation)
        y = linear(h, self.fc_out_weight, self.fc_out_bias)
        return y
