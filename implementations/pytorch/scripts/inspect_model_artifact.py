from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine_pt.weights.loader import WeightLoader


REQUIRED_GPT2_TENSORS = [
    "tok_embeddings.weight",
    "pos_embeddings.weight",
    "blocks.0.ln_1.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc_in.weight",
    "ln_f.weight",
    "lm_head.weight",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an engine-native model artifact.")
    parser.add_argument("--model-dir", required=True, help="Path to converted model directory")
    parser.add_argument(
        "--strict-gpt2",
        action="store_true",
        help="Verify a small set of required GPT-2 tensor names and shapes.",
    )
    args = parser.parse_args()

    loader = WeightLoader(args.model_dir)

    print("=== CONFIG ===")
    print(loader.config)
    print()

    print("=== METADATA ===")
    print(loader.dump_description_json())
    print()

    print("=== SAMPLE TENSORS ===")
    sample_names = loader.list_tensors()[: min(8, len(loader.list_tensors()))]
    for name in sample_names:
        entry = loader.get_entry(name)
        print(f"- {name}: shape={entry.shape}, dtype={entry.dtype}, offset={entry.offset}, nbytes={entry.nbytes}")
    print()

    if args.strict_gpt2:
        verify_gpt2_shapes(loader)
        print("GPT-2 artifact checks passed.")

    return 0


def verify_gpt2_shapes(loader: WeightLoader) -> None:
    cfg = loader.config

    expected_shapes = {
        "tok_embeddings.weight": (cfg.vocab_size, cfg.hidden_size),
        "pos_embeddings.weight": (cfg.max_position_embeddings, cfg.hidden_size),
        "blocks.0.ln_1.weight": (cfg.hidden_size,),
        "blocks.0.attn.qkv.weight": (3 * cfg.hidden_size, cfg.hidden_size),
        "blocks.0.attn.out_proj.weight": (cfg.hidden_size, cfg.hidden_size),
        "blocks.0.mlp.fc_in.weight": (cfg.intermediate_size, cfg.hidden_size),
        "ln_f.weight": (cfg.hidden_size,),
        "lm_head.weight": (cfg.vocab_size, cfg.hidden_size),
    }

    for name in REQUIRED_GPT2_TENSORS:
        if not loader.has_tensor(name):
            raise AssertionError(f"Required tensor missing: {name}")

    for name, expected_shape in expected_shapes.items():
        actual_shape = loader.get_entry(name).shape
        if actual_shape != expected_shape:
            raise AssertionError(
                f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
