from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from engine_pt.config.model_config import ModelConfig
from engine_pt.modules.attention import GPT2Attention
from engine_pt.modules.block import BlockOutput, GPT2Block
from engine_pt.modules.mlp import GPT2MLP
from engine_pt.ops.linear import linear
from engine_pt.ops.norms import layer_norm
from engine_pt.weights.loader import WeightLoader


@dataclass
class GPT2ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    presents: list[tuple[torch.Tensor | None, torch.Tensor | None]] | None = None


class GPT2Model:
    """Reference GPT-2 model built from engine-native artifacts.

    Phase 0 goals:
    - correctness and readability first
    - CPU-first, token-ids-first path
    - minimal dependencies beyond PyTorch
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        tok_embeddings_weight: torch.Tensor,
        pos_embeddings_weight: torch.Tensor,
        blocks: Sequence[GPT2Block],
        ln_f_weight: torch.Tensor,
        ln_f_bias: torch.Tensor | None,
        lm_head_weight: torch.Tensor,
    ) -> None:
        self.config = config
        self.tok_embeddings_weight = tok_embeddings_weight
        self.pos_embeddings_weight = pos_embeddings_weight
        self.blocks = list(blocks)
        self.ln_f_weight = ln_f_weight
        self.ln_f_bias = ln_f_bias
        self.lm_head_weight = lm_head_weight

        self._validate()

    def _validate(self) -> None:
        if self.config.model_family != "gpt2":
            raise ValueError(f"GPT2Model expects model_family='gpt2', got {self.config.model_family}")
        if len(self.blocks) != self.config.num_hidden_layers:
            raise ValueError(
                f"Expected {self.config.num_hidden_layers} blocks, got {len(self.blocks)}"
            )
        expected_tok = (self.config.vocab_size, self.config.hidden_size)
        if tuple(self.tok_embeddings_weight.shape) != expected_tok:
            raise ValueError(
                f"tok_embeddings.weight must have shape {expected_tok}, got {tuple(self.tok_embeddings_weight.shape)}"
            )
        expected_pos = (self.config.max_position_embeddings, self.config.hidden_size)
        if tuple(self.pos_embeddings_weight.shape) != expected_pos:
            raise ValueError(
                f"pos_embeddings.weight must have shape {expected_pos}, got {tuple(self.pos_embeddings_weight.shape)}"
            )
        expected_norm = (self.config.hidden_size,)
        if tuple(self.ln_f_weight.shape) != expected_norm:
            raise ValueError(
                f"ln_f.weight must have shape {expected_norm}, got {tuple(self.ln_f_weight.shape)}"
            )
        if self.ln_f_bias is not None and tuple(self.ln_f_bias.shape) != expected_norm:
            raise ValueError(
                f"ln_f.bias must have shape {expected_norm}, got {tuple(self.ln_f_bias.shape)}"
            )
        expected_lm_head = (self.config.vocab_size, self.config.hidden_size)
        if tuple(self.lm_head_weight.shape) != expected_lm_head:
            raise ValueError(
                f"lm_head.weight must have shape {expected_lm_head}, got {tuple(self.lm_head_weight.shape)}"
            )

    @classmethod
    def from_loader(
        cls,
        loader: WeightLoader,
        *,
        device: str | torch.device = "cpu",
    ) -> "GPT2Model":
        config = loader.config

        tok_embeddings_weight = loader.get("tok_embeddings.weight", device=device)
        pos_embeddings_weight = loader.get("pos_embeddings.weight", device=device)
        ln_f_weight = loader.get("ln_f.weight", device=device)
        ln_f_bias = loader.get("ln_f.bias", device=device) if loader.has_tensor("ln_f.bias") else None
        lm_head_weight = loader.get("lm_head.weight", device=device)

        blocks: list[GPT2Block] = []
        for i in range(config.num_hidden_layers):
            prefix = f"blocks.{i}"
            attn = GPT2Attention(
                config=config,
                qkv_weight=loader.get(f"{prefix}.attn.qkv.weight", device=device),
                qkv_bias=loader.get(f"{prefix}.attn.qkv.bias", device=device)
                if loader.has_tensor(f"{prefix}.attn.qkv.bias")
                else None,
                out_proj_weight=loader.get(f"{prefix}.attn.out_proj.weight", device=device),
                out_proj_bias=loader.get(f"{prefix}.attn.out_proj.bias", device=device)
                if loader.has_tensor(f"{prefix}.attn.out_proj.bias")
                else None,
            )
            mlp = GPT2MLP(
                config=config,
                fc_in_weight=loader.get(f"{prefix}.mlp.fc_in.weight", device=device),
                fc_in_bias=loader.get(f"{prefix}.mlp.fc_in.bias", device=device)
                if loader.has_tensor(f"{prefix}.mlp.fc_in.bias")
                else None,
                fc_out_weight=loader.get(f"{prefix}.mlp.fc_out.weight", device=device),
                fc_out_bias=loader.get(f"{prefix}.mlp.fc_out.bias", device=device)
                if loader.has_tensor(f"{prefix}.mlp.fc_out.bias")
                else None,
            )
            block = GPT2Block(
                config=config,
                ln_1_weight=loader.get(f"{prefix}.ln_1.weight", device=device),
                ln_1_bias=loader.get(f"{prefix}.ln_1.bias", device=device)
                if loader.has_tensor(f"{prefix}.ln_1.bias")
                else None,
                attention=attn,
                ln_2_weight=loader.get(f"{prefix}.ln_2.weight", device=device),
                ln_2_bias=loader.get(f"{prefix}.ln_2.bias", device=device)
                if loader.has_tensor(f"{prefix}.ln_2.bias")
                else None,
                mlp=mlp,
            )
            blocks.append(block)

        return cls(
            config=config,
            tok_embeddings_weight=tok_embeddings_weight,
            pos_embeddings_weight=pos_embeddings_weight,
            blocks=blocks,
            ln_f_weight=ln_f_weight,
            ln_f_bias=ln_f_bias,
            lm_head_weight=lm_head_weight,
        )

    def _embed(self, input_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"input_ids must be integer dtype, got {input_ids.dtype}")
        batch_size, seq_len = input_ids.shape
        if position_offset < 0:
            raise ValueError(f"position_offset must be non-negative, got {position_offset}")
        if position_offset + seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "Sequence positions "
                f"[{position_offset}, {position_offset + seq_len}) exceed max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        token_embeds = self.tok_embeddings_weight[input_ids]
        position_ids = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )
        position_embeds = self.pos_embeddings_weight[position_ids].unsqueeze(0)
        if tuple(position_embeds.shape) != (1, seq_len, self.config.hidden_size):
            raise RuntimeError(
                f"Unexpected position embedding shape {tuple(position_embeds.shape)}"
            )
        return token_embeds + position_embeds.expand(batch_size, -1, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: "KVCache | None" = None,
        use_cache: bool = False,
        return_hidden_states: bool = True,
    ) -> GPT2ModelOutput:
        position_offset = 0
        if cache is not None:
            cache_lengths = [cache.seq_len(layer_idx) for layer_idx in range(len(cache))]
            position_offset = cache_lengths[0] if cache_lengths else 0
            if any(length != position_offset for length in cache_lengths[1:]):
                raise ValueError(f"All cache layers must have the same sequence length, got {cache_lengths}")

        hidden_states = self._embed(input_ids, position_offset=position_offset)
        presents: list[tuple[torch.Tensor | None, torch.Tensor | None]] | None = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            past_k = None
            past_v = None
            if cache is not None:
                past_k, past_v = cache.get(layer_idx)
            block_out: BlockOutput = block.forward(
                hidden_states,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache,
            )
            hidden_states = block_out.hidden_states
            if presents is not None:
                presents.append((block_out.present_k, block_out.present_v))

        hidden_states = layer_norm(
            hidden_states,
            self.ln_f_weight,
            self.ln_f_bias,
            eps=self.config.norm_epsilon,
        )
        logits = linear(hidden_states, self.lm_head_weight, bias=None)

        return GPT2ModelOutput(
            logits=logits,
            hidden_states=hidden_states if return_hidden_states else torch.empty(0, device=logits.device),
            presents=presents,
        )

    def topk_last_token(self, input_ids: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(input_ids, use_cache=False, return_hidden_states=False)
        last_logits = out.logits[:, -1, :]
        return torch.topk(last_logits, k=k, dim=-1)
