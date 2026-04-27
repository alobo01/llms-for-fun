# Validation Strategy

## Purpose

Validation ensures that PyTorch, C++, and Rust implement the same engine semantics over the same artifacts.

Correctness is defined by a shared contract, not by informal similarity. Every implementation must consume the same converted model artifacts, the same token IDs or prompts, and the same test-vector definitions.

---

## Validation layers

Validation is split into two layers.

### 1. Model-math validation

Covers the path:

- input IDs
- embeddings
- transformer blocks
- final hidden state
- logits
- decode behavior

This is the primary correctness target for phase 0.

### 2. Tokenizer validation

Covers the path:

- prompt text to input IDs
- output IDs back to text

Tokenizer validation is important, but it is treated separately from model-math validation so that inference math bugs and tokenizer integration bugs do not block each other.

This is an explicit repository decision for phase 0.

---

## Reference implementation

The PyTorch engine is the correctness reference for phase 0.

Responsibilities of the PyTorch reference:

- load the converted GPT-2 artifact format
- run the forward pass
- run prefill and decode
- export golden test vectors
- provide the expected outputs used by cross-language tests

C++ and Rust are validated against the same artifacts and the same expected outputs.

---

## Test-vector strategy

Validation uses shared test vectors stored under `artifacts/test_vectors/`.

### Small smoke tests

JSON-based test vectors should include:

- human-readable name
- optional prompt text
- input IDs
- expected greedy next token
- optionally top logits for quick inspection
- tolerance policy

These are useful for:

- smoke testing
- CLI validation
- basic regression checks

### Tensor-rich tests

NPZ-based vectors should include larger arrays such as:

- input IDs
- hidden states after embeddings
- hidden states after one or more selected blocks
- final hidden state
- final logits
- prefill logits
- decode logits

These are useful for:

- debugging tensor mismatches
- narrowing down the first incorrect computation stage
- verifying cache behavior

---

## Required correctness checks

Every implementation must support the following checks.

### 1. Shape correctness

All outputs and intermediate tensors checked by a test vector must have the expected shape.

### 2. Dtype correctness

The runtime must use the expected dtype for model weights and checked outputs.

Phase 0 requires `float32` artifact support.

### 3. Logits agreement

Given the same input IDs and the same converted artifact, final-position logits must match the PyTorch reference within tolerance.

### 4. Greedy next-token agreement

For the same logits, greedy decoding must produce the exact same next token ID.

This check is integer-exact.

### 5. Multi-step generation agreement

For a fixed seed and fixed sampling parameters when applicable, generated token sequences should match the reference behavior under the shared CLI/runtime policy.

For deterministic greedy generation, exact token match is required.

### 6. Cache equivalence

This is one of the most important tests in the project.

For a prompt `P` and next token `t`, the following two procedures must agree for the final position logits within tolerance:

1. full recomputation on `P + t`
2. prefill on `P`, then one-step decode on `t` using the KV cache

If this test fails, the decode path is not trustworthy.

---

## Tolerance policy

Correctness is defined by numerical tolerance, not by bitwise identity.

Different implementations may differ slightly in floating-point accumulation order even when they are semantically correct.

### Phase 0 default tolerances

- hidden-state comparison: `atol = 1e-4`, `rtol = 1e-4`
- logits comparison: `atol = 1e-4`, `rtol = 1e-4`
- greedy next token: exact integer match

### Why this tolerance choice

This is the chosen phase-0 default because:

- all three implementations are CPU-first
- phase 0 artifact support is float32-only
- the target is stable cross-language validation without overfitting to a specific library's accumulation order

These tolerances may be tightened later if empirical results show that smaller tolerances are consistently achievable.

---

## Validation workflow

A recommended workflow is:

1. Convert GPT-2 reference weights into the engine artifact format.
2. Run the PyTorch reference on selected prompts or input IDs.
3. Export JSON and NPZ golden vectors.
4. Run the same cases in C++ and Rust.
5. Compare outputs using a shared comparison tool.
6. Record regressions under `tests/regression/` or equivalent.

This workflow ensures that every implementation is tested against the same source artifacts and expected outputs.

---

## Debugging order when a test fails

When a cross-language mismatch appears, debug in this order:

1. confirm config fields and tensor shapes
2. confirm weight-name mapping and tensor loading
3. confirm embedding outputs
4. confirm block-by-block hidden-state agreement
5. confirm masking and attention score path
6. confirm cache append semantics and position handling
7. confirm final logits and sampling behavior

This order usually finds errors earlier and more cheaply than debugging the generation loop first.

---

## Benchmarking is not validation

Benchmark results may reveal bugs, but they are not a substitute for correctness checks.

A fast engine that fails cache equivalence or produces unstable logits is not correct.

Validation must be completed before performance claims are treated as meaningful.

---

## Explicit phase-0 decisions

### Decision 1: separate tokenizer and model-math validation

This is the right choice for phase 0 because it reduces ambiguity when debugging and allows engine-math work to proceed even if tokenizer integration differs temporarily across languages.

### Decision 2: PyTorch is the golden-vector producer

This is the right choice because Python is the easiest environment for checkpoint conversion, tensor inspection, and generation of rich debugging artifacts. It also avoids duplicating tooling logic in C++ and Rust.

### Decision 3: cache equivalence is mandatory, not optional

This is the right choice because cache correctness is central to any inference engine. If the decode path cannot be validated against full recomputation, later phases will rest on an unstable foundation.
