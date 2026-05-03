from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def numpy_dtype_for(dtype: str) -> np.dtype:
    try:
        return np.dtype(DTYPE_TO_NUMPY[dtype])
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for phase 0 converter: {dtype}") from exc


def contiguous_array(array: np.ndarray, dtype: str) -> np.ndarray:
    np_dtype = numpy_dtype_for(dtype)
    return np.ascontiguousarray(array.astype(np_dtype, copy=False))


def nbytes_for_shape(shape: Iterable[int], dtype: str) -> int:
    np_dtype = numpy_dtype_for(dtype)
    count = 1
    for dim in shape:
        count *= int(dim)
    return int(count * np_dtype.itemsize)
