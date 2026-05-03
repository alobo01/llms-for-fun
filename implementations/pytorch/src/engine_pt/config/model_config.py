from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Internal engine model configuration.

    This mirrors the repository's engine-native config schema rather than any
    external framework config. The goal is to keep runtime code independent of
    Hugging Face field naming and make later model-family support easier.
    """

    model_family: str
    model_name: str
    architecture: str
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    activation: str
    norm_type: str
    norm_epsilon: float
    positional_encoding: str
    rope_theta: float | None
    rope_scaling: dict[str, Any] | None
    mlp_type: str
    attention_type: str
    attention_bias: bool
    mlp_bias: bool
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_id: int | None
    initializer_range: float | None
    dtype: str

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_decoder_only(self) -> bool:
        return self.architecture == "decoder_only"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "norm_type": self.norm_type,
            "norm_epsilon": self.norm_epsilon,
            "positional_encoding": self.positional_encoding,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "mlp_type": self.mlp_type,
            "attention_type": self.attention_type,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "initializer_range": self.initializer_range,
            "dtype": self.dtype,
        }

    def validate(self) -> None:
        if not self.is_decoder_only:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} vs {self.num_key_value_heads}"
            )
        if self.positional_encoding not in {"learned_absolute", "rope"}:
            raise ValueError(f"Unsupported positional_encoding: {self.positional_encoding}")
        if self.norm_type not in {"layer_norm", "rms_norm"}:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        if self.attention_type not in {"mha", "mqa", "gqa"}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")
        if self.mlp_type not in {"gelu_ffn", "swiglu", "geglu"}:
            raise ValueError(f"Unsupported mlp_type: {self.mlp_type}")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        if self.positional_encoding == "learned_absolute" and self.rope_theta is not None:
            raise ValueError("rope_theta must be None for learned_absolute positional encoding")

    def __str__(self) -> str:
        fields = [
            f"model_family={self.model_family}",
            f"model_name={self.model_name}",
            f"hidden_size={self.hidden_size}",
            f"num_hidden_layers={self.num_hidden_layers}",
            f"num_attention_heads={self.num_attention_heads}",
            f"num_key_value_heads={self.num_key_value_heads}",
            f"head_dim={self.head_dim}",
            f"intermediate_size={self.intermediate_size}",
            f"dtype={self.dtype}",
        ]
        return f"ModelConfig({', '.join(fields)})"
