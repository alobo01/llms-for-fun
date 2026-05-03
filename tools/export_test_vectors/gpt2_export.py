from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch


def _ensure_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pytorch_src = repo_root / "implementations" / "pytorch" / "src"
    for path in (str(pytorch_src), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_import_paths()

from engine_pt.weights.loader import WeightLoader
from engine_pt.models.gpt2.model import GPT2Model
from engine_pt.generation.prefill import prefill
from engine_pt.generation.decode import decode_step


DEFAULT_SMOKE_CASES = [
    {"name": "smoke_hello_world", "input_ids": [15496, 995], "appended_token_id": 11},
    {"name": "smoke_short_repeat", "input_ids": [464, 464, 464], "appended_token_id": 11},
]


def _topk_logits(logits_1d: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
    values, indices = torch.topk(logits_1d, k=min(k, logits_1d.shape[-1]))
    return [
        {"token_id": int(idx.item()), "logit": float(val.item())}
        for val, idx in zip(values, indices)
    ]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_case(model: GPT2Model, output_dir: Path, name: str, input_ids_list: List[int], appended_token_id: int, atol: float, rtol: float) -> None:
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    appended = torch.tensor([[appended_token_id]], dtype=torch.long)

    with torch.no_grad():
        full_out = model.forward(torch.cat([input_ids, appended], dim=1), use_cache=False, return_hidden_states=True)
        prefill_out = prefill(model, input_ids)
        decode_out = decode_step(model, appended, prefill_out.cache)

        # Intermediate tensors
        emb = model._embed(input_ids)
        block0_out = model.blocks[0].forward(emb)

    final_logits = full_out.logits[0, -1].detach().cpu()
    prefill_logits = prefill_out.logits.detach().cpu().numpy()
    decode_logits = decode_out.logits.detach().cpu().numpy()

    smoke = {
        "name": name,
        "input_ids": input_ids_list,
        "appended_token_id": appended_token_id,
        "expected_next_token_greedy": int(torch.argmax(prefill_out.logits[0, -1]).item()),
        "logits_last_token_top5": _topk_logits(prefill_out.logits[0, -1], 5),
        "tolerance": {"atol": atol, "rtol": rtol},
        "dtype": str(prefill_out.logits.dtype).replace("torch.", ""),
    }

    with (output_dir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2)

    np.savez(
        output_dir / f"{name}.npz",
        input_ids=input_ids.detach().cpu().numpy(),
        appended_token_id=np.array([appended_token_id], dtype=np.int64),
        hidden_after_embeddings=emb.detach().cpu().numpy(),
        hidden_after_block_0=block0_out.hidden_states.detach().cpu().numpy(),
        final_hidden=full_out.hidden_states.detach().cpu().numpy(),
        final_logits=full_out.logits.detach().cpu().numpy(),
        prefill_logits=prefill_logits,
        decode_logits=decode_logits,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Export golden GPT-2 test vectors for later C++/Rust validation.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/test_vectors/gpt2/gpt2")
    _ensure_dir(output_dir)

    loader = WeightLoader(model_dir)
    model = GPT2Model.from_loader(loader)

    manifest = {
        "format_version": 1,
        "model_family": "gpt2",
        "model_name": loader.config.model_name,
        "cases": [],
    }

    for case in DEFAULT_SMOKE_CASES:
        export_case(
            model=model,
            output_dir=output_dir,
            name=case["name"],
            input_ids_list=case["input_ids"],
            appended_token_id=case["appended_token_id"],
            atol=args.atol,
            rtol=args.rtol,
        )
        manifest["cases"].append({
            "name": case["name"],
            "json": f"{case['name']}.json",
            "npz": f"{case['name']}.npz",
        })

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== GPT-2 VECTOR EXPORT ===")
    print(f"model_dir: {model_dir}")
    print(f"output_dir: {output_dir}")
    print(f"exported_cases: {len(manifest['cases'])}")
    for case in manifest["cases"]:
        print(f"- {case['name']}")


if __name__ == "__main__":
    main()
