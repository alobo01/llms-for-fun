# Tensor Naming Specification

## Purpose

This document defines the canonical internal tensor names used by the inference engine after checkpoint conversion. These names are the stable contract consumed by the PyTorch, C++, and Rust implementations.

The main goal is to decouple engine internals from upstream checkpoint naming conventions such as Hugging Face field names. A conversion tool is responsible for mapping external names into the internal naming scheme defined here.

---

## Design principles

The tensor naming scheme follows these rules:

- names must be stable across languages
- names must describe logical role rather than framework-specific implementation details
- names must be independent of external checkpoint formats
- names must generalize cleanly to later model families
- names must be readable enough to debug with simple tooling

The engine should treat these names as the authoritative identifiers for tensors in converted artifacts.

---

## Naming conventions

### General form

Tensor names use dot-separated logical paths.

Examples:

```text
blocks.0.attn.qkv.weight
blocks.7.mlp.fc_out.bias
ln_f.weight
```

### Reserved top-level prefixes

The following top-level prefixes are reserved:

- `tok_embeddings`
- `pos_embeddings`
- `blocks`
- `ln_f`
- `lm_head`

Future model families may introduce additional top-level names if needed, but these must be documented in a family-specific extension.

### Layer indexing

Transformer blocks are indexed from zero using the path:

```text
blocks.{i}.
```

where `{i}` is the zero-based layer index.

### Parameter suffixes

The standard parameter suffixes are:

- `weight`
- `bias`

Phase 0 only requires these two suffixes.

---

## Canonical GPT-2 tensor names

The required GPT-2 tensor names are:

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

---

## GPT-2 logical meaning of each tensor

### Embeddings

- `tok_embeddings.weight`
  - token embedding table
  - shape: `[vocab_size, hidden_size]`

- `pos_embeddings.weight`
  - learned absolute position embedding table
  - shape: `[max_position_embeddings, hidden_size]`

### Per-block tensors

For each transformer block `i`:

- `blocks.{i}.ln_1.weight`
- `blocks.{i}.ln_1.bias`
  - pre-attention layer norm parameters
  - shape: `[hidden_size]`

- `blocks.{i}.attn.qkv.weight`
- `blocks.{i}.attn.qkv.bias`
  - fused projection that produces query, key, and value activations
  - weight shape: `[3 * hidden_size, hidden_size]`
  - bias shape: `[3 * hidden_size]`

- `blocks.{i}.attn.out_proj.weight`
- `blocks.{i}.attn.out_proj.bias`
  - attention output projection
  - weight shape: `[hidden_size, hidden_size]`
  - bias shape: `[hidden_size]`

- `blocks.{i}.ln_2.weight`
- `blocks.{i}.ln_2.bias`
  - pre-MLP layer norm parameters
  - shape: `[hidden_size]`

- `blocks.{i}.mlp.fc_in.weight`
- `blocks.{i}.mlp.fc_in.bias`
  - MLP input projection
  - weight shape: `[intermediate_size, hidden_size]`
  - bias shape: `[intermediate_size]`

- `blocks.{i}.mlp.fc_out.weight`
- `blocks.{i}.mlp.fc_out.bias`
  - MLP output projection
  - weight shape: `[hidden_size, intermediate_size]`
  - bias shape: `[hidden_size]`

### Final tensors

- `ln_f.weight`
- `ln_f.bias`
  - final layer norm parameters
  - shape: `[hidden_size]`

- `lm_head.weight`
  - output projection to vocabulary logits
  - shape: `[vocab_size, hidden_size]`

---

## Shape and layout conventions

All shapes in the artifact index are recorded in row-major contiguous order.

The naming specification does not require a specific internal in-memory layout during execution, but the meaning of each dimension must remain consistent with the artifact metadata.

For phase 0:

- dense weight matrices are stored as rank-2 tensors
- bias vectors are stored as rank-1 tensors
- layer norm parameters are stored as rank-1 tensors

---

## Fused and unfused representation policy

### Phase 0 decision

For GPT-2, attention input projection is stored as a single fused tensor pair:

- `blocks.{i}.attn.qkv.weight`
- `blocks.{i}.attn.qkv.bias`

instead of separate `q_proj`, `k_proj`, and `v_proj` tensors.

### Justification

This matches the common GPT-2 checkpoint structure closely, reduces conversion complexity in phase 0, and still keeps the logical role explicit enough for later adapters.

Later model families may use separate names such as:

- `blocks.{i}.attn.q_proj.weight`
- `blocks.{i}.attn.k_proj.weight`
- `blocks.{i}.attn.v_proj.weight`

if their checkpoint structure or engine implementation makes that cleaner. Those would be family-specific extensions rather than changes to the GPT-2 contract.

---

## Tied embeddings policy

### Phase 0 decision

`lm_head.weight` must appear explicitly in the artifact index even when embeddings are tied.

### Justification

This keeps the artifact self-describing and simplifies loaders in PyTorch, C++, and Rust. A loader may internally alias `lm_head.weight` to `tok_embeddings.weight` when `tie_word_embeddings = true`, but the artifact format itself does not require alias semantics in phase 0.

---

## Conversion policy

A checkpoint conversion tool must:

1. load the upstream checkpoint
2. map upstream tensor names to canonical internal names
3. validate expected shapes
4. emit the canonical names into `weights.index.json`
5. write the raw tensor bytes into `weights.bin`

The engine must not depend on upstream names after conversion is complete.

---

## Validation rules

A valid converted GPT-2 artifact must satisfy all of the following:

- all required tensor names are present
- no duplicate tensor names exist
- every tensor listed in the index has a valid shape and dtype
- `lm_head.weight` exists explicitly
- per-layer tensor counts match `num_hidden_layers`
- attention projection shapes match `hidden_size`
- MLP shapes match `intermediate_size`

---

## Future extension guidance

This naming scheme is intentionally simple enough to extend.

Examples of later additions include:

- separate Q, K, V projections
- RMSNorm weights without bias
- gated MLP tensors such as `gate_proj`, `up_proj`, `down_proj`
- rotary-embedding-related parameters if a family stores them explicitly

When extending the scheme, prefer adding clearly named logical tensors rather than changing existing GPT-2 names.
