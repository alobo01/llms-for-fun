from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import sys

import torch

PYTORCH_SRC = Path(__file__).resolve().parents[2]
if str(PYTORCH_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_SRC))
    
from engine_pt.weights.loader import WeightLoader
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.generation.prefill import prefill
from engine_pt.generation.decode import decode_step


@dataclass(frozen=True)
class CacheEquivalenceResult:
    input_ids: list[int]
    appended_token_id: int
    full_logits_shape: tuple[int, ...]
    decode_logits_shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    rms_diff: float
    allclose: bool
    greedy_match: bool
    atol: float
    rtol: float


def _parse_ids(text: str) -> list[int]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one token id in --input-ids")
    return [int(v) for v in values]


def run_cache_equivalence(
    model_dir: str | Path,
    input_ids: Sequence[int],
    appended_token_id: int,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    device: str = "cpu",
) -> CacheEquivalenceResult:
    loader = WeightLoader(model_dir)
    model = GPT2Model.from_loader(loader, device=device)

    prompt = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    appended = torch.tensor([[int(appended_token_id)]], dtype=torch.long, device=device)
    full_input = torch.cat([prompt, appended], dim=1)

    with torch.no_grad():
        full_out = model.forward(full_input, use_cache=False)
        prefill_out = prefill(model, prompt)
        decode_out = decode_step(model, appended, prefill_out.cache)

    full_last = full_out.logits[:, -1, :].detach().cpu()
    decode_last = decode_out.logits[:, -1, :].detach().cpu()

    diff = (full_last - decode_last).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    rms_diff = float(torch.sqrt(((full_last - decode_last) ** 2).mean()).item())
    allclose = bool(torch.allclose(full_last, decode_last, atol=atol, rtol=rtol))
    greedy_match = bool(int(full_last.argmax(dim=-1).item()) == int(decode_last.argmax(dim=-1).item()))

    return CacheEquivalenceResult(
        input_ids=list(input_ids),
        appended_token_id=int(appended_token_id),
        full_logits_shape=tuple(full_last.shape),
        decode_logits_shape=tuple(decode_last.shape),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rms_diff=rms_diff,
        allclose=allclose,
        greedy_match=greedy_match,
        atol=float(atol),
        rtol=float(rtol),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-2 cache equivalence check.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-ids", required=True, help="Comma-separated prompt token ids.")
    parser.add_argument("--appended-token-id", required=True, type=int, help="Single token appended after prefill.")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    result = run_cache_equivalence(
        model_dir=args.model_dir,
        input_ids=_parse_ids(args.input_ids),
        appended_token_id=args.appended_token_id,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("=== CACHE EQUIVALENCE ===")
    print(f"input_ids: {result.input_ids}")
    print(f"appended_token_id: {result.appended_token_id}")
    print(f"full last logits shape: {result.full_logits_shape}")
    print(f"decode logits shape:    {result.decode_logits_shape}")
    print(f"max_abs_diff: {result.max_abs_diff:.8f}")
    print(f"mean_abs_diff: {result.mean_abs_diff:.8f}")
    print(f"rms_diff: {result.rms_diff:.8f}")
    print(f"allclose: {result.allclose} (atol={result.atol}, rtol={result.rtol})")
    print(f"greedy_match: {result.greedy_match}")

    if not (result.allclose and result.greedy_match):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
