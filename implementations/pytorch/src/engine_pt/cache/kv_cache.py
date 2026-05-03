from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from engine_pt.config.model_config import ModelConfig


@dataclass
class LayerKV:
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None


class KVCache:
    """Append-only KV cache for decoder-only models.

    Canonical logical layout per layer:
        keys:   [batch, kv_heads, seq, head_dim]
        values: [batch, kv_heads, seq, head_dim]
    """

    def __init__(self, layers: List[LayerKV], max_seq_len: Optional[int] = None) -> None:
        self.layers = layers
        self.max_seq_len = max_seq_len

    @classmethod
    def empty_from_config(
        cls,
        config: ModelConfig,
        max_seq_len: Optional[int] = None,
    ) -> "KVCache":
        return cls([LayerKV() for _ in range(config.num_hidden_layers)], max_seq_len=max_seq_len)

    def __len__(self) -> int:
        return len(self.layers)

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def update(self, layer_idx: int, present_k: torch.Tensor, present_v: torch.Tensor) -> None:
        if present_k.ndim != 4 or present_v.ndim != 4:
            raise ValueError("present_k and present_v must have shape [B, kv_heads, seq, head_dim]")
        if present_k.shape != present_v.shape:
            raise ValueError(
                f"present_k and present_v shapes must match, got {tuple(present_k.shape)} vs {tuple(present_v.shape)}"
            )
        if self.max_seq_len is not None and present_k.shape[2] > self.max_seq_len:
            raise ValueError(
                f"cache sequence length {present_k.shape[2]} exceeds max_seq_len={self.max_seq_len}"
            )
        self.layers[layer_idx] = LayerKV(keys=present_k, values=present_v)

    def seq_len(self, layer_idx: int = 0) -> int:
        k, _ = self.get(layer_idx)
        return 0 if k is None else int(k.shape[2])

    def clone(self) -> "KVCache":
        new_layers: List[LayerKV] = []
        for layer in self.layers:
            new_layers.append(
                LayerKV(
                    keys=None if layer.keys is None else layer.keys.clone(),
                    values=None if layer.values is None else layer.values.clone(),
                )
            )
        return KVCache(new_layers, max_seq_len=self.max_seq_len)
