from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


@dataclass(frozen=True)
class WeightMappingRule:
    source_name: str
    target_name: str


class GPT2WeightMapper:
    """Maps Hugging Face GPT-2 state dict names to engine-internal tensor names.

    Phase 0 uses a fused QKV tensor name: `blocks.{i}.attn.qkv.*`.
    """

    @staticmethod
    def build_name_map(num_hidden_layers: int) -> dict[str, str]:
        mapping: dict[str, str] = {
            "transformer.wte.weight": "tok_embeddings.weight",
            "transformer.wpe.weight": "pos_embeddings.weight",
            "transformer.ln_f.weight": "ln_f.weight",
            "transformer.ln_f.bias": "ln_f.bias",
            "lm_head.weight": "lm_head.weight",
        }

        for i in range(num_hidden_layers):
            prefix = f"transformer.h.{i}"
            target = f"blocks.{i}"
            mapping.update(
                {
                    f"{prefix}.ln_1.weight": f"{target}.ln_1.weight",
                    f"{prefix}.ln_1.bias": f"{target}.ln_1.bias",
                    f"{prefix}.attn.c_attn.weight": f"{target}.attn.qkv.weight",
                    f"{prefix}.attn.c_attn.bias": f"{target}.attn.qkv.bias",
                    f"{prefix}.attn.c_proj.weight": f"{target}.attn.out_proj.weight",
                    f"{prefix}.attn.c_proj.bias": f"{target}.attn.out_proj.bias",
                    f"{prefix}.ln_2.weight": f"{target}.ln_2.weight",
                    f"{prefix}.ln_2.bias": f"{target}.ln_2.bias",
                    f"{prefix}.mlp.c_fc.weight": f"{target}.mlp.fc_in.weight",
                    f"{prefix}.mlp.c_fc.bias": f"{target}.mlp.fc_in.bias",
                    f"{prefix}.mlp.c_proj.weight": f"{target}.mlp.fc_out.weight",
                    f"{prefix}.mlp.c_proj.bias": f"{target}.mlp.fc_out.bias",
                }
            )
        return mapping

    @classmethod
    def map_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        num_hidden_layers: int,
        include_lm_head_if_missing: bool = True,
    ) -> dict[str, torch.Tensor]:
        name_map = cls.build_name_map(num_hidden_layers)
        converted: dict[str, torch.Tensor] = {}

        for source_name, tensor in state_dict.items():
            if source_name not in name_map:
                continue
            target_name = name_map[source_name]
            converted[target_name] = cls._transform_tensor(source_name, target_name, tensor)

        if include_lm_head_if_missing and "lm_head.weight" not in converted:
            if "tok_embeddings.weight" not in converted:
                raise KeyError("tok_embeddings.weight missing; cannot synthesize lm_head.weight")
            converted["lm_head.weight"] = converted["tok_embeddings.weight"]

        cls._validate_required(converted, num_hidden_layers)
        return converted

    @staticmethod
    def _transform_tensor(source_name: str, target_name: str, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().cpu().contiguous()

        # Hugging Face Conv1D weights are stored as [in_features, out_features].
        # The engine uses linear weights in [out_features, in_features].
        if source_name.endswith(("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
            out = out.t().contiguous()
        return out

    @staticmethod
    def _validate_required(converted: dict[str, torch.Tensor], num_hidden_layers: int) -> None:
        required = {
            "tok_embeddings.weight",
            "pos_embeddings.weight",
            "ln_f.weight",
            "ln_f.bias",
            "lm_head.weight",
        }
        for i in range(num_hidden_layers):
            required.update(
                {
                    f"blocks.{i}.ln_1.weight",
                    f"blocks.{i}.ln_1.bias",
                    f"blocks.{i}.attn.qkv.weight",
                    f"blocks.{i}.attn.qkv.bias",
                    f"blocks.{i}.attn.out_proj.weight",
                    f"blocks.{i}.attn.out_proj.bias",
                    f"blocks.{i}.ln_2.weight",
                    f"blocks.{i}.ln_2.bias",
                    f"blocks.{i}.mlp.fc_in.weight",
                    f"blocks.{i}.mlp.fc_in.bias",
                    f"blocks.{i}.mlp.fc_out.weight",
                    f"blocks.{i}.mlp.fc_out.bias",
                }
            )
        missing = sorted(required - set(converted))
        if missing:
            raise KeyError(f"Converted state dict missing required tensors: {missing[:8]}{'...' if len(missing) > 8 else ''}")
