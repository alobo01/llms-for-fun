from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_SRC = REPO_ROOT / "implementations" / "pytorch" / "src"
TOOLS_DIR = REPO_ROOT / "tools"
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from engine_pt.models.gpt2.weight_mapper import GPT2WeightMapper
from convert_weights.common import contiguous_array, ensure_parent, write_json


DEFAULT_MODEL_ID = "openai-community/gpt2"


def build_engine_config(hf_config: Any, *, dtype: str) -> dict[str, Any]:
    return {
        "model_family": "gpt2",
        "model_name": getattr(hf_config, "_name_or_path", DEFAULT_MODEL_ID).split("/")[-1] or "gpt2",
        "architecture": "decoder_only",
        "vocab_size": int(hf_config.vocab_size),
        "max_position_embeddings": int(hf_config.n_positions),
        "hidden_size": int(hf_config.n_embd),
        "num_hidden_layers": int(hf_config.n_layer),
        "num_attention_heads": int(hf_config.n_head),
        "num_key_value_heads": int(hf_config.n_head),
        "intermediate_size": int(hf_config.n_inner or (4 * hf_config.n_embd)),
        "activation": str(getattr(hf_config, "activation_function", "gelu_new")),
        "norm_type": "layer_norm",
        "norm_epsilon": float(hf_config.layer_norm_epsilon),
        "positional_encoding": "learned_absolute",
        "rope_theta": None,
        "rope_scaling": None,
        "mlp_type": "gelu_ffn",
        "attention_type": "mha",
        "attention_bias": True,
        "mlp_bias": True,
        "tie_word_embeddings": bool(getattr(hf_config, "tie_word_embeddings", True)),
        "bos_token_id": int(hf_config.bos_token_id),
        "eos_token_id": int(hf_config.eos_token_id),
        "initializer_range": float(hf_config.initializer_range),
        "dtype": dtype,
    }


def write_engine_checkpoint(
    output_dir: Path,
    *,
    engine_config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    dtype: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.bin"
    index_path = output_dir / "weights.index.json"
    metadata_path = output_dir / "metadata.json"
    config_path = output_dir / "config.json"

    index: dict[str, Any] = {}
    offset = 0

    with weights_path.open("wb") as f:
        for name in sorted(tensors):
            arr = tensors[name].detach().cpu().numpy()
            arr = contiguous_array(arr, dtype)
            f.write(arr.tobytes(order="C"))
            nbytes = int(arr.nbytes)
            index[name] = {
                "dtype": dtype,
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": nbytes,
            }
            offset += nbytes

    metadata = {
        "format_version": 1,
        "model_family": engine_config["model_family"],
        "model_name": engine_config["model_name"],
        "endianness": "little",
        "default_dtype": dtype,
        "tensor_count": len(index),
        "weight_file": weights_path.name,
        "index_file": index_path.name,
    }

    write_json(config_path, engine_config)
    write_json(metadata_path, metadata)
    write_json(index_path, index)


def convert(model_id: str, output_dir: Path, dtype: str) -> None:
    hf_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    engine_config = build_engine_config(hf_config, dtype=dtype)
    converted = GPT2WeightMapper.map_state_dict(
        model.state_dict(),
        num_hidden_layers=engine_config["num_hidden_layers"],
        include_lm_head_if_missing=True,
    )
    write_engine_checkpoint(output_dir, engine_config=engine_config, tensors=converted, dtype=dtype)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face GPT-2 checkpoint into engine artifact format.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local directory containing a GPT-2 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "models" / "gpt2" / "gpt2" / "converted"),
        help="Directory where config.json, metadata.json, weights.index.json, and weights.bin will be written.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Storage dtype for phase 0 converted weights.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    convert(args.model_id, output_dir, args.dtype)
    print(f"Wrote converted GPT-2 artifact to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
