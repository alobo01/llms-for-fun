from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTORCH_SRC = REPO_ROOT / "implementations" / "pytorch" / "src"
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))

from engine_pt.weights.loader import WeightLoader
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.generation.prefill import prefill
from engine_pt.generation.decode import decode_step


def _parse_ids(text: str) -> list[int]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one token id in --input-ids")
    return [int(v) for v in values]


def _topk_pairs(logits: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return [
        {"token_id": int(idx.item()), "logit": float(val.item())}
        for val, idx in zip(values, indices)
    ]


def run_cache_smoke(
    model_dir: str | Path,
    input_ids: Sequence[int],
    appended_token_id: int,
    *,
    topk: int = 5,
    device: str = "cpu",
) -> dict:
    loader = WeightLoader(model_dir)
    model = GPT2Model.from_loader(loader, device=device)

    prompt = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    appended = torch.tensor([[int(appended_token_id)]], dtype=torch.long, device=device)

    with torch.no_grad():
        prefill_out = prefill(model, prompt)
        decode_out = decode_step(model, appended, prefill_out.cache)

    final_prefill_logits = prefill_out.logits[:, -1, :].detach().cpu()
    final_decode_logits = decode_out.logits[:, -1, :].detach().cpu()

    cache_lengths = [prefill_out.cache.seq_len(i) for i in range(model.config.num_hidden_layers)]
    decode_cache_lengths = [decode_out.cache.seq_len(i) for i in range(model.config.num_hidden_layers)]

    return {
        "model_dir": str(model_dir),
        "input_ids": list(input_ids),
        "appended_token_id": int(appended_token_id),
        "prefill_logits_shape": list(prefill_out.logits.shape),
        "decode_logits_shape": list(decode_out.logits.shape),
        "hidden_states_shape": list(decode_out.hidden_states.shape),
        "prefill_cache_seq_lens": cache_lengths,
        "decode_cache_seq_lens": decode_cache_lengths,
        "prefill_last_top_logits": _topk_pairs(final_prefill_logits[0], topk),
        "decode_top_logits": _topk_pairs(final_decode_logits[0], topk),
        "decode_greedy_next_token": int(final_decode_logits.argmax(dim=-1).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GPT-2 cache smoke test.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-ids", required=True, help="Comma-separated prompt token ids.")
    parser.add_argument("--appended-token-id", required=True, type=int)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_cache_smoke(
        model_dir=args.model_dir,
        input_ids=_parse_ids(args.input_ids),
        appended_token_id=args.appended_token_id,
        topk=args.topk,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=== GPT-2 CACHE SMOKE ===")
    print(f"model_dir: {result['model_dir']}")
    print(f"input_ids: {result['input_ids']}")
    print(f"appended_token_id: {result['appended_token_id']}")
    print(f"prefill logits shape: {tuple(result['prefill_logits_shape'])}")
    print(f"decode logits shape:  {tuple(result['decode_logits_shape'])}")
    print(f"hidden_states shape:  {tuple(result['hidden_states_shape'])}")
    print(f"prefill cache seq lens (first 4): {result['prefill_cache_seq_lens'][:4]}")
    print(f"decode cache seq lens (first 4):  {result['decode_cache_seq_lens'][:4]}")
    print("top logits for decode step:")
    for row in result["decode_top_logits"]:
        print(f"  token_id={row['token_id']:<6} logit={row['logit']:.6f}")
    print(f"decode greedy next token: {result['decode_greedy_next_token']}")


if __name__ == "__main__":
    main()
