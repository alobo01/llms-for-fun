from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


_DTYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


@dataclass(frozen=True)
class TensorIndexEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    source_name: str | None = None
    checksum: str | None = None
    notes: str | None = None

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    @property
    def expected_nbytes(self) -> int:
        return self.numel * dtype_size(self.dtype)

    def validate(self) -> None:
        if self.dtype not in _DTYPE_SIZES:
            raise ValueError(f"Unsupported dtype for tensor {self.name}: {self.dtype}")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid shape for tensor {self.name}: {self.shape}")
        if self.offset < 0:
            raise ValueError(f"Negative offset for tensor {self.name}: {self.offset}")
        if self.nbytes <= 0:
            raise ValueError(f"Non-positive nbytes for tensor {self.name}: {self.nbytes}")
        if self.nbytes != self.expected_nbytes:
            raise ValueError(
                f"nbytes mismatch for tensor {self.name}: got {self.nbytes}, "
                f"expected {self.expected_nbytes} from shape={self.shape} dtype={self.dtype}"
            )


@dataclass(frozen=True)
class WeightIndex:
    entries: dict[str, TensorIndexEntry]

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WeightIndex":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)

        entries: dict[str, TensorIndexEntry] = {}
        for name, entry in raw.items():
            parsed = TensorIndexEntry(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(int(x) for x in entry["shape"]),
                offset=int(entry["offset"]),
                nbytes=int(entry["nbytes"]),
                source_name=entry.get("source_name"),
                checksum=entry.get("checksum"),
                notes=entry.get("notes"),
            )
            parsed.validate()
            entries[name] = parsed

        return cls(entries=entries)

    def __contains__(self, tensor_name: str) -> bool:
        return tensor_name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> list[str]:
        return sorted(self.entries.keys())

    def get(self, tensor_name: str) -> TensorIndexEntry:
        try:
            return self.entries[tensor_name]
        except KeyError as exc:
            available = ", ".join(self.names()[:10])
            suffix = "..." if len(self.entries) > 10 else ""
            raise KeyError(
                f"Tensor {tensor_name!r} not found in weight index. "
                f"Available names include: {available}{suffix}"
            ) from exc


@dataclass(frozen=True)
class ArtifactMetadata:
    format_version: int
    model_family: str
    model_name: str
    endianness: str
    default_dtype: str
    tensor_count: int
    weight_file: str
    index_file: str

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ArtifactMetadata":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        metadata = cls(**data)
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError(f"Unsupported format_version: {self.format_version}")
        if self.endianness != "little":
            raise ValueError(f"Unsupported endianness: {self.endianness}")
        if self.default_dtype not in _DTYPE_SIZES:
            raise ValueError(f"Unsupported default_dtype: {self.default_dtype}")
        if self.tensor_count <= 0:
            raise ValueError("tensor_count must be positive")


def dtype_size(dtype: str) -> int:
    try:
        return _DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc
