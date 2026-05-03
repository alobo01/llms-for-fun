from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import torch

from engine_pt.config.model_config import ModelConfig
from engine_pt.weights.index import ArtifactMetadata, WeightIndex


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class WeightLoader:
    """Loads engine-native model artifacts.

    Phase 0 design goals:
    - simple and explicit loading
    - shape-aware and spec-driven
    - easy to debug from Python before C++ and Rust ports exist
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig.from_json_file(self.model_dir / "config.json")
        self.metadata = ArtifactMetadata.from_json_file(self.model_dir / "metadata.json")
        self.index = WeightIndex.from_json_file(self.model_dir / self.metadata.index_file)
        self.weight_file = self.model_dir / self.metadata.weight_file

        if not self.weight_file.exists():
            raise FileNotFoundError(f"weights.bin not found: {self.weight_file}")

        if len(self.index) != self.metadata.tensor_count:
            raise ValueError(
                f"tensor_count mismatch: metadata says {self.metadata.tensor_count}, "
                f"index contains {len(self.index)} entries"
            )

        self._validate_file_bounds()

    def _validate_file_bounds(self) -> None:
        file_size = self.weight_file.stat().st_size
        for entry in self.index.entries.values():
            end = entry.offset + entry.nbytes
            if end > file_size:
                raise ValueError(
                    f"Tensor {entry.name} extends beyond weights.bin: "
                    f"end={end}, file_size={file_size}"
                )

    def list_tensors(self) -> list[str]:
        return self.index.names()

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self.index

    def get_entry(self, tensor_name: str):
        return self.index.get(tensor_name)

    def get(self, tensor_name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        entry = self.index.get(tensor_name)
        torch_dtype = self._torch_dtype(entry.dtype)

        with self.weight_file.open("rb") as f:
            f.seek(entry.offset)
            raw = f.read(entry.nbytes)

        tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).clone()
        return tensor.reshape(entry.shape).to(device)

    def load_many(
        self,
        tensor_names: Iterable[str],
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        return {name: self.get(name, device=device) for name in tensor_names}

    def describe(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "config": self.config.to_dict(),
            "metadata": {
                "format_version": self.metadata.format_version,
                "model_family": self.metadata.model_family,
                "model_name": self.metadata.model_name,
                "endianness": self.metadata.endianness,
                "default_dtype": self.metadata.default_dtype,
                "tensor_count": self.metadata.tensor_count,
                "weight_file": self.metadata.weight_file,
                "index_file": self.metadata.index_file,
            },
        }

    def dump_description_json(self) -> str:
        return json.dumps(self.describe(), indent=2)

    @staticmethod
    def _torch_dtype(dtype: str) -> torch.dtype:
        try:
            return _TORCH_DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported tensor dtype: {dtype}") from exc
