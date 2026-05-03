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

from engine_pt.generation.generator import SamplingConfig, generate
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.weights.loader import WeightLoader


def parse_input_ids(value: str) -> list[int]:
    value = value.strip()
    if not value:
        raise ValueError("--input-ids must not be empty")
    return [int(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-2 generation smoke test.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    loader = WeightLoader(args.model_dir)
    model = GPT2Model.from_loader(loader)
    input_ids = torch.tensor([parse_input_ids(args.input_ids)], dtype=torch.long)
    sampling = SamplingConfig(
        mode=args.mode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    out = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=model.config.eos_token_id,
        sampling=sampling,
    )

    payload = {
        "input_ids": input_ids[0].tolist(),
        "generated_ids": out.generated_ids[0].tolist(),
        "full_sequence_ids": out.full_sequence_ids[0].tolist(),
        "steps": out.steps,
        "stopped_on_eos": out.stopped_on_eos,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=== GENERATION SMOKE ===")
        print(f"input_ids: {payload['input_ids']}")
        print(f"generated_ids: {payload['generated_ids']}")
        print(f"full_sequence_ids: {payload['full_sequence_ids']}")
        print(f"steps: {payload['steps']}")
        print(f"stopped_on_eos: {payload['stopped_on_eos']}")
