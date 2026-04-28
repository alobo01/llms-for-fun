# Weight Artifact Format

## Purpose

This document defines the engine-native weight artifact format consumed by PyTorch, C++, and Rust implementations.

The format is intentionally decoupled from framework-specific checkpoint serialization. A conversion tool is responsible for mapping reference checkpoints into this format.

---

## Phase-0 decision

Phase 0 uses a simple artifact layout:

```text
<model_dir>/
  config.json
  metadata.json
  weights.index.json
  weights.bin
```

This is the chosen format because it is:

- simple to implement in all three languages
- easy to inspect and debug
- friendly to later memory mapping
- independent of PyTorch or Hugging Face serialization internals

---

## Files

## `config.json`

Contains the model configuration using the internal engine schema defined in `specs/model_config_schema.json`.

This file is the logical description of the model.

## `metadata.json`

Contains artifact-level metadata.

Recommended fields:

```json
{
  "format_version": 1,
  "model_family": "gpt2",
  "model_name": "gpt2",
  "endianness": "little",
  "default_dtype": "float32",
  "tensor_count": 29,
  "weight_file": "weights.bin",
  "index_file": "weights.index.json"
}
```

### Required metadata fields

- `format_version`
- `model_family`
- `model_name`
- `endianness`
- `default_dtype`
- `tensor_count`
- `weight_file`
- `index_file`

## `weights.index.json`

Maps internal tensor names to location and shape metadata.

Example:

```json
{
  "tok_embeddings.weight": {
    "dtype": "float32",
    "shape": [50257, 768],
    "offset": 0,
    "nbytes": 154389504
  },
  "pos_embeddings.weight": {
    "dtype": "float32",
    "shape": [1024, 768],
    "offset": 154389504,
    "nbytes": 3145728
  }
}
```

### Required per-tensor fields

- `dtype`
- `shape`
- `offset`
- `nbytes`

### Optional per-tensor fields

- `source_name`
- `checksum`
- `notes`

Optional fields may assist debugging, but loaders must not depend on them for correctness.

## `weights.bin`

A raw binary file containing tensor data packed one tensor after another. Tensor byte ranges are defined by each tensor's explicit `offset` and `nbytes` fields in `weights.index.json`; loaders must not rely on JSON object key order.

Rules:

- little-endian
- contiguous row-major storage
- no compression in phase 0
- no framework-specific headers

---

## Internal tensor names

Loaders must use internal tensor names, not external checkpoint names.

For GPT-2, the canonical internal tensor names are:

```text
tok_embeddings.weight
pos_embeddings.weight
blocks.{i}.ln_1.weight
blocks.{i}.ln_1.bias
blocks.{i}.attn.qkv.weight
blocks.{i}.attn.qkv.bias
blocks.{i}.attn.out_proj.weight
blocks.{i}.attn.out_proj.bias
blocks.{i}.ln_2.weight
blocks.{i}.ln_2.bias
blocks.{i}.mlp.fc_in.weight
blocks.{i}.mlp.fc_in.bias
blocks.{i}.mlp.fc_out.weight
blocks.{i}.mlp.fc_out.bias
ln_f.weight
ln_f.bias
lm_head.weight
```

A conversion tool is responsible for mapping external source names into this namespace.

---

## GPT-2 tensor shapes

For GPT-2, the expected logical shapes are:

- `tok_embeddings.weight`: `[vocab_size, hidden_size]`
- `pos_embeddings.weight`: `[max_position_embeddings, hidden_size]`
- `blocks.{i}.ln_1.weight`: `[hidden_size]`
- `blocks.{i}.ln_1.bias`: `[hidden_size]`
- `blocks.{i}.attn.qkv.weight`: `[3 * hidden_size, hidden_size]`
- `blocks.{i}.attn.qkv.bias`: `[3 * hidden_size]`
- `blocks.{i}.attn.out_proj.weight`: `[hidden_size, hidden_size]`
- `blocks.{i}.attn.out_proj.bias`: `[hidden_size]`
- `blocks.{i}.ln_2.weight`: `[hidden_size]`
- `blocks.{i}.ln_2.bias`: `[hidden_size]`
- `blocks.{i}.mlp.fc_in.weight`: `[intermediate_size, hidden_size]`
- `blocks.{i}.mlp.fc_in.bias`: `[intermediate_size]`
- `blocks.{i}.mlp.fc_out.weight`: `[hidden_size, intermediate_size]`
- `blocks.{i}.mlp.fc_out.bias`: `[hidden_size]`
- `ln_f.weight`: `[hidden_size]`
- `ln_f.bias`: `[hidden_size]`
- `lm_head.weight`: `[vocab_size, hidden_size]`

---

## Explicit decision: `lm_head.weight` is indexed explicitly

Even when `tie_word_embeddings = true`, phase 0 requires `lm_head.weight` to exist as an explicit entry in `weights.index.json`.

This is the chosen design because:

- it keeps the artifact format uniform
- it avoids requiring alias semantics in every loader immediately
- it makes artifact inspection and debugging easier

Implementations may choose to alias `lm_head.weight` to `tok_embeddings.weight` internally after loading when the config indicates tied embeddings, but that is an optimization step and not a file-format rule.

---

## Supported dtypes

### Required in phase 0

- `float32`

### Reserved for later

- `float16`
- `bfloat16`

The config schema reserves these dtype names for later phases, but phase-0 artifact compliance only requires `float32`. Loaders may reject unsupported dtypes cleanly.

---

## Validation requirements

A valid artifact must satisfy:

1. `config.json` conforms to `specs/model_config_schema.json`.
2. `metadata.json` names the correct weight and index files.
3. every tensor in `weights.index.json` has a valid offset and byte range within `weights.bin`.
4. tensor byte size matches `shape × dtype_size`.
5. all required tensor names for the target model family are present.

---

## Conversion responsibilities

The checkpoint conversion tool must:

- read the source checkpoint
- map source tensor names to internal tensor names
- ensure tensors are contiguous in the chosen storage order
- write `config.json`
- write `metadata.json`
- write `weights.index.json`
- write `weights.bin`

The conversion tool must not leak source-framework naming into runtime loading logic.

---

## Future evolution

This format is intentionally simple for phase 0.

Later phases may add:

- memory-mapped loading guidance
- quantized tensor records
- sharded weight files
- optional checksums or stronger integrity metadata

Any such change should increase `format_version` and preserve backward compatibility where feasible.
