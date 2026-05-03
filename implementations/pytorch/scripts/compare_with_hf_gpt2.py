from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import sys

import torch
from transformers import GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTORCH_SRC = REPO_ROOT / "implementations" / "pytorch" / "src"
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))
    
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.weights.loader import WeightLoader


def _parse_input_ids(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--input-ids must contain at least one integer token id")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid --input-ids value: {raw!r}") from exc


def _topk_summary(logits_1d: torch.Tensor, k: int) -> list[dict[str, Any]]:
    k = max(1, min(k, logits_1d.numel()))
    values, indices = torch.topk(logits_1d, k=k)
    return [
        {"token_id": int(idx.item()), "logit": float(val.item())}
        for val, idx in zip(values, indices)
    ]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare engine GPT-2 logits with Hugging Face GPT-2")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to converted engine artifact directory",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="openai-community/gpt2",
        help="Hugging Face GPT-2 model id or local checkpoint path",
    )
    parser.add_argument(
        "--input-ids",
        type=str,
        required=True,
        help="Comma-separated token ids, e.g. 15496,995",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top logits to print from the final position",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for logits parity check",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for logits parity check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of pretty text",
    )
    return parser


def _compare_logits(engine_logits: torch.Tensor, hf_logits: torch.Tensor) -> dict[str, Any]:
    diff = (engine_logits - hf_logits).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms = float(torch.sqrt(torch.mean((engine_logits - hf_logits) ** 2)).item())
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rms_diff": rms,
    }


def main() -> None:
    args = _build_argparser().parse_args()
    input_ids_list = _parse_input_ids(args.input_ids)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    loader = WeightLoader(args.model_dir)
    engine_model = GPT2Model.from_loader(loader)

    hf_model = GPT2LMHeadModel.from_pretrained(args.hf_model_id)
    hf_model.eval()

    with torch.no_grad():
        engine_out = engine_model.forward(input_ids)
        hf_out = hf_model(input_ids=input_ids)

    engine_logits = engine_out.logits.detach().cpu()
    hf_logits = hf_out.logits.detach().cpu()

    if engine_logits.shape != hf_logits.shape:
        raise RuntimeError(
            f"Shape mismatch: engine logits {tuple(engine_logits.shape)} vs HF logits {tuple(hf_logits.shape)}"
        )

    metrics = _compare_logits(engine_logits, hf_logits)
    allclose_ok = bool(torch.allclose(engine_logits, hf_logits, atol=args.atol, rtol=args.rtol))

    engine_last = engine_logits[0, -1]
    hf_last = hf_logits[0, -1]
    engine_next = int(torch.argmax(engine_last).item())
    hf_next = int(torch.argmax(hf_last).item())

    result = {
        "model_dir": str(args.model_dir),
        "hf_model_id": args.hf_model_id,
        "input_ids": input_ids_list,
        "engine_logits_shape": list(engine_logits.shape),
        "hf_logits_shape": list(hf_logits.shape),
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "allclose": allclose_ok,
        "greedy_match": engine_next == hf_next,
        "engine_next_token": engine_next,
        "hf_next_token": hf_next,
        "metrics": metrics,
        "engine_top_logits_final": _topk_summary(engine_last, args.top_k),
        "hf_top_logits_final": _topk_summary(hf_last, args.top_k),
        "config": asdict(engine_model.config),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("=== GPT-2 HF PARITY CHECK ===")
    print(f"model_dir: {args.model_dir}")
    print(f"hf_model_id: {args.hf_model_id}")
    print(f"input_ids: {input_ids_list}")
    print(f"engine logits shape: {tuple(engine_logits.shape)}")
    print(f"hf logits shape: {tuple(hf_logits.shape)}")
    print(f"tolerance: atol={args.atol} rtol={args.rtol}")
    print()
    print("metrics:")
    print(f"  max_abs_diff:  {metrics['max_abs_diff']:.8f}")
    print(f"  mean_abs_diff: {metrics['mean_abs_diff']:.8f}")
    print(f"  rms_diff:      {metrics['rms_diff']:.8f}")
    print(f"  allclose:      {allclose_ok}")
    print(f"  greedy_match:  {engine_next == hf_next}")
    print(f"  engine next token: {engine_next}")
    print(f"  hf next token:     {hf_next}")
    print()
    print("engine top logits for final position:")
    for item in result["engine_top_logits_final"]:
        print(f"  token_id={item['token_id']:<6} logit={item['logit']:.6f}")
    print()
    print("HF top logits for final position:")
    for item in result["hf_top_logits_final"]:
        print(f"  token_id={item['token_id']:<6} logit={item['logit']:.6f}")


if __name__ == "__main__":
    main()
