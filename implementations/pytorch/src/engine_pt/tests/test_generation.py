from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from typing import Any
from pathlib import Path
import sys

import torch

PYTORCH_SRC = Path(__file__).resolve().parents[2]
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))

from engine_pt.generation.generator import SamplingConfig, generate
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.weights.loader import WeightLoader


def _parse_ids(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _run_greedy(model: GPT2Model, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    x = torch.tensor([input_ids], dtype=torch.long)
    out = generate(
        model=model,
        input_ids=x,
        sampling=SamplingConfig(mode="greedy"),
        max_new_tokens=max_new_tokens,
        eos_token_id=model.config.eos_token_id,
    )
    return {
        "generated_ids": out.generated_ids.squeeze(0).tolist(),
        "full_sequence_ids": out.full_sequence_ids.squeeze(0).tolist(),
        "steps": out.steps,
        "stopped_on_eos": out.stopped_on_eos,
    }


def _run_sample(model: GPT2Model, input_ids: list[int], max_new_tokens: int, seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    x = torch.tensor([input_ids], dtype=torch.long)
    out = generate(
        model=model,
        input_ids=x,
        sampling=SamplingConfig(mode="sample", temperature=0.8, top_k=50, top_p=0.95),
        max_new_tokens=max_new_tokens,
        eos_token_id=model.config.eos_token_id,
    )
    return {
        "generated_ids": out.generated_ids.squeeze(0).tolist(),
        "full_sequence_ids": out.full_sequence_ids.squeeze(0).tolist(),
        "steps": out.steps,
        "stopped_on_eos": out.stopped_on_eos,
    }


def run_tests(model_dir: str, input_ids: list[int], max_new_tokens: int, verbose: bool = False) -> dict[str, Any]:
    loader = WeightLoader(model_dir)
    model = GPT2Model.from_loader(loader)

    results: dict[str, Any] = {"model_dir": model_dir, "input_ids": input_ids, "max_new_tokens": max_new_tokens, "tests": {}}

    # Greedy determinism
    greedy_a = _run_greedy(model, input_ids, max_new_tokens)
    greedy_b = _run_greedy(model, input_ids, max_new_tokens)
    greedy_ok = greedy_a == greedy_b
    results["tests"]["greedy_determinism"] = {
        "passed": greedy_ok,
        "run_a": greedy_a,
        "run_b": greedy_b,
    }

    # Sample determinism with seed
    sample_a = _run_sample(model, input_ids, max_new_tokens, seed=123)
    sample_b = _run_sample(model, input_ids, max_new_tokens, seed=123)
    sample_ok = sample_a == sample_b
    results["tests"]["sample_determinism_seeded"] = {
        "passed": sample_ok,
        "run_a": sample_a,
        "run_b": sample_b,
    }

    # Length invariants
    gen_len = len(greedy_a["generated_ids"])
    full_len = len(greedy_a["full_sequence_ids"])
    len_ok = gen_len <= max_new_tokens and full_len == len(input_ids) + gen_len and greedy_a["steps"] == gen_len
    results["tests"]["length_invariants"] = {
        "passed": len_ok,
        "generated_len": gen_len,
        "full_len": full_len,
        "prompt_len": len(input_ids),
        "steps": greedy_a["steps"],
    }

    # EOS stopping behavior (invariant-only)
    eos_id = model.config.eos_token_id
    eos_ok = True
    if greedy_a["stopped_on_eos"]:
        eos_ok = len(greedy_a["generated_ids"]) > 0 and greedy_a["generated_ids"][-1] == eos_id
    results["tests"]["eos_stopping"] = {
        "passed": eos_ok,
        "stopped_on_eos": greedy_a["stopped_on_eos"],
        "eos_token_id": eos_id,
        "last_generated_id": greedy_a["generated_ids"][-1] if greedy_a["generated_ids"] else None,
    }

    all_passed = all(t["passed"] for t in results["tests"].values())
    results["all_passed"] = all_passed

    if verbose:
        print(json.dumps(results, indent=2))
    else:
        print("=== GENERATION TESTS ===")
        print(f"model_dir: {model_dir}")
        print(f"input_ids: {input_ids}")
        for name, payload in results["tests"].items():
            print(f"- {name}: {'PASS' if payload['passed'] else 'FAIL'}")
        print(f"all_passed: {all_passed}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase 0 GPT-2 generation tests.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-ids", default="15496,995")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_tests(
        model_dir=args.model_dir,
        input_ids=_parse_ids(args.input_ids),
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose and not args.json,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
