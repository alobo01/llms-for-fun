from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTORCH_SRC = REPO_ROOT / "implementations" / "pytorch" / "src"
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))

from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.weights.loader import WeightLoader


def parse_input_ids(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--input-ids must contain at least one token id")
    return [int(p) for p in parts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GPT-2 smoke forward pass from engine artifacts.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to converted model artifact directory",
    )
    parser.add_argument(
        "--input-ids",
        type=str,
        default="15496,995",
        help="Comma-separated token ids. Default corresponds to a small GPT-2 smoke prompt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top logits to print for the final token position",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_ids_list = parse_input_ids(args.input_ids)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    loader = WeightLoader(args.model_dir)
    model = GPT2Model.from_loader(loader)

    output = model.forward(input_ids)
    last_logits = output.logits[0, -1]
    top_values, top_indices = torch.topk(last_logits, k=args.top_k, dim=-1)

    result = {
        "model_dir": str(args.model_dir),
        "config": {
            "model_family": loader.config.model_family,
            "model_name": loader.config.model_name,
            "hidden_size": loader.config.hidden_size,
            "num_hidden_layers": loader.config.num_hidden_layers,
            "num_attention_heads": loader.config.num_attention_heads,
        },
        "input_ids": input_ids_list,
        "logits_shape": list(output.logits.shape),
        "hidden_states_shape": list(output.hidden_states.shape),
        "top_logits": [
            {"token_id": int(idx), "logit": float(val)}
            for idx, val in zip(top_indices.tolist(), top_values.tolist())
        ],
        "next_token_greedy": int(top_indices[0].item()),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=== GPT-2 SMOKE RUN ===")
    print(f"model_dir: {args.model_dir}")
    print(f"input_ids: {input_ids_list}")
    print(f"logits shape: {tuple(output.logits.shape)}")
    print(f"hidden_states shape: {tuple(output.hidden_states.shape)}")
    print("top logits for final position:")
    for item in result["top_logits"]:
        print(f"  token_id={item['token_id']:<6} logit={item['logit']:.6f}")
    print(f"greedy next token: {result['next_token_greedy']}")


if __name__ == "__main__":
    main()
