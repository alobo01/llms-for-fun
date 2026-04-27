# Test Vector Format Specification

## Purpose

This document defines the shared format for correctness test vectors used by the inference engine. Test vectors are produced by the PyTorch reference implementation and consumed by PyTorch, C++, and Rust validation workflows.

The main goal is to make correctness testing language-independent. All implementations should be able to consume the same prompts, token IDs, expected outputs, and tolerance settings.

---

## Design principles

The test vector format follows these rules:

- small smoke tests should be easy to read and diff
- tensor-rich debugging artifacts should support larger arrays efficiently
- the format should separate model-math correctness from tokenizer correctness
- tolerances must be explicit
- cache equivalence must be testable using shared artifacts

---

## File types

Phase 0 uses two complementary formats:

- `JSON` for human-readable smoke and regression cases
- `NPZ` for intermediate tensors and larger numeric outputs

### Phase 0 decision

Use JSON for small end-to-end cases and NPZ for dense tensor dumps.

### Justification

JSON is ideal for smoke tests, CI diffs, and manual inspection. NPZ is much better for arrays such as hidden states and logits, especially when exporting multiple intermediate tensors from the PyTorch reference.

---

## Directory layout

Test vectors should be stored under:

```text
artifacts/test_vectors/<family>/<model_name>/
```

For GPT-2 phase 0, the recommended path is:

```text
artifacts/test_vectors/gpt2/gpt2/
```

Example contents:

```text
artifacts/test_vectors/gpt2/gpt2/
  smoke_hello.json
  smoke_paragraph.json
  cache_equivalence_case_01.json
  blockwise_case_01.npz
  final_logits_case_01.npz
```

---

## JSON smoke test format

JSON test vectors are intended for:

- short prompts
- top-k logit checks
- greedy token checks
- sampling configuration checks
- cache equivalence metadata

### Required fields

A valid JSON smoke test must include:

- `name`
- `model_family`
- `model_name`
- `input_ids`
- `dtype`
- `tolerance`

### Optional fields

A JSON smoke test may include:

- `prompt_text`
- `expected_next_token_greedy`
- `expected_generated_ids`
- `logits_last_token_topk`
- `sampling`
- `notes`

### Example

```json
{
  "name": "smoke_hello",
  "model_family": "gpt2",
  "model_name": "gpt2",
  "prompt_text": "Hello world",
  "input_ids": [15496, 995],
  "dtype": "float32",
  "tolerance": {
    "atol": 1e-4,
    "rtol": 1e-4
  },
  "expected_next_token_greedy": 0,
  "logits_last_token_topk": [
    {"token_id": 0, "logit": 11.23},
    {"token_id": 13, "logit": 10.91},
    {"token_id": 11, "logit": 10.44}
  ]
}
```

---

## NPZ tensor test format

NPZ test vectors are intended for:

- intermediate hidden states
- final hidden states
- full or sliced logits
- cache equivalence tensor comparisons
- regression debugging

### Required arrays

There is no single mandatory set of arrays for all NPZ files, but every NPZ file must include enough metadata to describe what it contains.

### Required metadata sidecar

Each NPZ file must have a JSON sidecar with the same basename.

Example:

```text
blockwise_case_01.npz
blockwise_case_01.json
```

The sidecar must include:

- `name`
- `model_family`
- `model_name`
- `dtype`
- `tolerance`
- `arrays`

### Example sidecar

```json
{
  "name": "blockwise_case_01",
  "model_family": "gpt2",
  "model_name": "gpt2",
  "dtype": "float32",
  "tolerance": {
    "atol": 1e-4,
    "rtol": 1e-4
  },
  "arrays": [
    "input_ids",
    "hidden_after_embeddings",
    "hidden_after_block_0",
    "final_hidden",
    "final_logits"
  ]
}
```

### Recommended array names

The following array names are recommended for GPT-2 phase 0:

- `input_ids`
- `position_ids`
- `hidden_after_embeddings`
- `hidden_after_block_0`
- `hidden_after_block_N`
- `final_hidden`
- `final_logits`
- `prefill_logits`
- `decode_logits`
- `greedy_generated_ids`

Not every file needs all of them.

---

## Tokenization policy

### Phase 0 decision

Model correctness vectors must treat token IDs as the primary contract. Raw text is optional metadata.

### Justification

This isolates model-math correctness from tokenizer implementation issues. End-to-end tokenizer parity can be tested separately, but the core inference engine should first be validated from `input_ids` to outputs.

### Rule

- `input_ids` are authoritative
- `prompt_text` is informative only

If `prompt_text` and `input_ids` disagree, the test should be considered malformed.

---

## Tolerance specification

Every test vector must carry explicit tolerance settings.

Required structure:

```json
{
  "tolerance": {
    "atol": 1e-4,
    "rtol": 1e-4
  }
}
```

### Phase 0 decision

Use default float32 correctness tolerances of:

- `atol = 1e-4`
- `rtol = 1e-4`

### Justification

This is a practical starting point for CPU-first cross-language validation where accumulation order may differ slightly but implementations should still agree tightly.

---

## Cache equivalence test format

Cache equivalence is a required correctness class in phase 0.

A cache equivalence test should encode:

- `prompt_input_ids`
- `next_input_ids`
- `prefill_expected`
- `decode_expected`
- `comparison_target`

### Example JSON

```json
{
  "name": "cache_equivalence_case_01",
  "model_family": "gpt2",
  "model_name": "gpt2",
  "prompt_input_ids": [15496, 995],
  "next_input_ids": [11],
  "dtype": "float32",
  "tolerance": {
    "atol": 1e-4,
    "rtol": 1e-4
  },
  "comparison_target": "final_position_logits"
}
```

### Required correctness rule

For the same prompt and next token:

- full recomputation on `prompt + next token`
- prefill on `prompt` followed by decode on `next token`

must match within tolerance for the chosen comparison target.

---

## Top-k logit checks

For smoke tests, JSON files may store only the top-k logits for the final position rather than the entire logits tensor.

### Phase 0 decision

Use top-k logit summaries in JSON smoke tests and full logits in NPZ files when needed.

### Justification

This keeps human-readable tests small while preserving access to full numeric detail in dedicated regression artifacts.

---

## Sampling metadata

If a test covers stochastic sampling rather than greedy decode, it must include explicit sampling parameters.

Example:

```json
{
  "sampling": {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "seed": 1234
  }
}
```

### Rule

- Greedy tests do not require `sampling`
- Stochastic tests must specify a seed if deterministic replay is expected

---

## Validation rules

A valid test vector must satisfy the following:

- `model_family` and `model_name` are present
- `dtype` is present
- `tolerance` is present
- token ID arrays are integer-valued
- numeric arrays match the declared dtype
- sidecar metadata matches NPZ array keys exactly
- required comparison target exists for cache equivalence cases

---

## Recommended test classes for phase 0

The PyTorch reference should export at least these classes of tests:

1. Smoke next-token test
2. Multi-token greedy generation test
3. Cache equivalence test
4. Intermediate tensor regression test
5. Final logits regression test

This set is enough to validate the core GPT-2 engine before adding later model families.
