from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "tools" / "export_test_vectors" / "gpt2_export.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from test_export_vectors.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the GPT-2 golden vector exporter.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/test_vectors/gpt2/gpt2")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).parent)
    exporter = repo_root / "tools" / "export_test_vectors" / "gpt2_export.py"
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(exporter),
        "--model-dir",
        args.model_dir,
        "--output-dir",
        str(output_dir),
    ]
    env = dict(**__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root / 'implementations' / 'pytorch' / 'src'}:{repo_root}:{existing}" if existing else f"{repo_root / 'implementations' / 'pytorch' / 'src'}:{repo_root}"
    subprocess.run(cmd, check=True, env=env)

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not created"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest.get("cases", [])) >= 1, "no exported cases found"

    for case in manifest["cases"]:
        json_path = output_dir / case["json"]
        npz_path = output_dir / case["npz"]
        assert json_path.exists(), f"missing JSON vector: {json_path.name}"
        assert npz_path.exists(), f"missing NPZ vector: {npz_path.name}"

        smoke = json.loads(json_path.read_text(encoding="utf-8"))
        assert "input_ids" in smoke and len(smoke["input_ids"]) >= 1
        assert "expected_next_token_greedy" in smoke
        assert "logits_last_token_top5" in smoke and len(smoke["logits_last_token_top5"]) >= 1

        data = np.load(npz_path)
        required = {
            "input_ids",
            "appended_token_id",
            "hidden_after_embeddings",
            "hidden_after_block_0",
            "final_hidden",
            "final_logits",
            "prefill_logits",
            "decode_logits",
        }
        missing = required.difference(set(data.files))
        assert not missing, f"missing arrays in {npz_path.name}: {sorted(missing)}"

    print("=== EXPORT VECTOR TEST ===")
    print(f"model_dir: {args.model_dir}")
    print(f"output_dir: {output_dir}")
    print(f"cases: {len(manifest['cases'])}")
    print("all_passed: True")


if __name__ == "__main__":
    main()
