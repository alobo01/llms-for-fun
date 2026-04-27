# Phase 0 — GPT-2

## Purpose

Phase 0 establishes the first complete end-to-end inference engine implementation. The initial target model family is GPT-2, chosen as a baseline dense decoder-only transformer that is simple enough to validate and rich enough to exercise the full runtime.

The goal of phase 0 is not just to make GPT-2 generate text. The real goal is to build the engine foundations that later phases can extend without large structural rewrites.

---

## Why GPT-2 is the first target

GPT-2 is a good first target because it includes the essential decoder-only runtime path while avoiding many modern architectural complications.

It exercises:

- token embeddings
- learned absolute position embeddings
- masked self-attention
- MLP blocks
- LayerNorm
- residual connections
- logits projection
- autoregressive generation
- cache-based incremental decode

This makes GPT-2 large enough to be meaningful and small enough to keep milestone 0 focused.

---

## Phase 0 scope

Phase 0 supports a CPU-first GPT-2 inference engine in:

- PyTorch
- C++
- Rust

### Required capabilities

1. Load GPT-2 weights converted into the engine artifact format.
2. Accept token IDs as model input.
3. Run a full-sequence forward pass.
4. Run autoregressive decode using a KV cache.
5. Produce next-token logits.
6. Generate tokens with:
   - greedy decoding
   - temperature
   - top-k
   - top-p
7. Validate correctness against a PyTorch reference.
8. Support shared benchmark prompts and benchmark harness inputs.

### Required runtime split

Phase 0 must explicitly separate:

- prefill
- decode

This is required both for correctness testing and for later performance evaluation.

---

## Non-goals

Phase 0 does not require:

- GPU kernels
- quantization
- speculative decoding
- paged attention
- sliding-window attention
- MQA or GQA behavior beyond the future-proof config field
- support for non-GPT-2 families
- distributed or parallel inference across devices

These may become later milestones, but they are not part of phase 0.

---

## Deliverables

Phase 0 should produce the following.

### 1. Shared artifact pipeline

A conversion tool that maps GPT-2 reference checkpoints into the engine artifact format.

Expected outputs:

- `config.json`
- `metadata.json`
- `weights.index.json`
- `weights.bin`
- tokenizer assets

### 2. Shared specs and docs

Stable documentation and specs for:

- architecture
- weight format
- cache layout
- test-vector format
- validation policy

### 3. PyTorch reference engine

The PyTorch implementation is the first complete reference.

It should:

- load converted GPT-2 artifacts
- run the forward pass
- run cache-based decode
- export golden vectors
- serve as the primary correctness oracle

### 4. C++ engine

A C++ implementation that consumes the same artifacts and matches PyTorch behavior within tolerance.

### 5. Rust engine

A Rust implementation that consumes the same artifacts and matches PyTorch behavior within tolerance.

### 6. Cross-language validation harness

A shared validation path that compares:

- shapes
- selected hidden states
- logits
- greedy next-token outputs
- cache equivalence behavior

### 7. Baseline benchmark harness

At minimum, phase 0 should support measurement of:

- prefill latency
- decode latency per token
- end-to-end generation latency
- small-batch throughput

---

## Milestone breakdown

## Milestone 0.1 — specs and artifact format

Define the contracts used by every later implementation.

Deliver:

- `docs/architecture.md`
- `docs/phase0-gpt2.md`
- `docs/validation.md`
- `specs/model_config_schema.json`
- `specs/weight_format.md`
- `specs/cache_layout.md`

## Milestone 0.2 — PyTorch reference correctness

Implement the first reference GPT-2 engine in PyTorch.

Deliver:

- config loader
- weight loader
- GPT-2 forward pass
- append-only KV cache
- prefill and decode paths
- greedy generation
- golden test-vector exporter

## Milestone 0.3 — C++ correctness engine

Implement the same engine behavior in C++.

Deliver:

- same artifact consumption
- same logical runtime behavior
- matching smoke-test outputs within tolerance

## Milestone 0.4 — Rust correctness engine

Implement the same engine behavior in Rust.

Deliver:

- same artifact consumption
- same logical runtime behavior
- matching smoke-test outputs within tolerance

## Milestone 0.5 — unified validation

Create cross-language test runners and regression checks.

Deliver:

- shared smoke tests
- cache equivalence tests
- generation parity checks

## Milestone 0.6 — benchmark baseline

Create the first comparable benchmark suite.

Deliver:

- benchmark prompt sets
- benchmark configs
- reporting scripts
- first profiling summaries

---

## Core engineering decisions for phase 0

### 1. GPT-2 is implemented through future-proof interfaces

Although GPT-2 only requires:

- learned absolute positions
- LayerNorm
- GELU-style MLP
- standard multi-head attention

phase 0 config and loader interfaces already include fields for:

- `num_key_value_heads`
- `positional_encoding`
- `norm_type`
- `mlp_type`
- `attention_type`

This allows later support for LLaMA-style, Gemma, Mistral, and Qwen-like families without changing the core config contract.

### 2. Cache layout is standardized now

Phase 0 defines canonical logical KV-cache layout as:

- keys: `[batch, kv_heads, seq, head_dim]`
- values: `[batch, kv_heads, seq, head_dim]`

For GPT-2, `kv_heads == attention_heads`, but the chosen layout generalizes cleanly to MQA and GQA later.

### 3. Model correctness is separated from tokenizer correctness

Phase 0 validates the model from token IDs to logits and generated token IDs independently of full tokenizer parity. This keeps debugging focused and makes the multi-language rollout more manageable.

---

## Success criteria

Phase 0 is successful when all three implementations can:

1. Load the same GPT-2 converted artifacts.
2. Produce matching final-position logits within tolerance.
3. Produce identical greedy next-token outputs.
4. Pass cache equivalence tests.
5. Run the same benchmark prompts through a shared benchmark interface.

The most important outcome is not only a working GPT-2 engine, but a repository structure and contract set that makes later model-family support incremental.

---

## What phase 0 should make easy later

By the end of phase 0, the project should be ready to add later model families mainly by introducing new combinations of:

- positional encoding strategy
- norm type
- MLP type
- attention head layout
- cache policy
- checkpoint mapping

That is the real preparation for modern realism.
