# Architecture Overview

## Purpose

This repository defines the architecture and shared contracts for a decoder-only language model inference engine intended to be implemented in three languages: PyTorch, C++, and Rust. The project begins with GPT-2 support and evolves through a staged roadmap toward modern dense open LLM families.

The repository is organized around shared engine contracts rather than language-specific implementations. Shared specs define model config schema, tensor naming, weight artifact format, cache layout, test-vector formats, and CLI behavior. Language implementations consume those specs and artifacts while remaining free to differ in internal optimization strategy.

The design goal is that adding support for a new model family should mostly mean adding a new model adapter and a few new block variants, not redesigning the whole engine.

---

## Design principles

### 1. Shared contracts first

The source of truth for repository-wide behavior lives in `specs/`.

Rules:

- `specs/` defines contracts.
- `artifacts/` contains data that follows those contracts.
- `implementations/` consume artifacts without redefining the format.
- `tools/` may generate or validate artifacts, but may not silently invent incompatible formats.

### 2. Engine core separated from model-family support

The repository should distinguish between:

- reusable engine subsystems
- model-family-specific wiring

Reusable engine subsystems include:

- config parsing
- weight loading
- tokenizer integration
- tensor and buffer handling
- numeric ops
- transformer modules
- KV cache management
- sampling
- generation loop
- validation and benchmarking

Model-family-specific code should be limited to:

- checkpoint mapping into internal tensor names
- layer/block structure
- position handling strategy
- norm and MLP variants
- attention head layout details

### 3. CPU-first and correctness-first

Phase 0 is CPU-first. The initial target is not peak performance, but a correct and extensible runtime.

This is intentional. Early GPU support would shift effort toward backend-specific kernel work before the engine contracts are stable.

### 4. Prefill and decode are first-class paths

The engine must distinguish between:

- prefill: process an entire prompt sequence and build cache state
- decode: process one or more incremental generation steps using cached state

This distinction must exist from phase 0 onward because it affects correctness testing, cache design, and benchmarking.

### 5. Future-proofed abstractions from phase 0

Although GPT-2 is the first model family, core abstractions must already leave room for later support of:

- learned absolute positions and RoPE
- LayerNorm and RMSNorm
- GELU MLPs and gated MLPs such as SwiGLU or GeGLU
- MHA, MQA, and GQA
- append-only and sliding-window cache policies

That future-proofing should happen at the config and interface level, not by prematurely implementing every feature.

---

## Repository structure

A representative target repository layout is shown below. During milestone 0.1 the repository may contain only `docs/` and `specs/`; implementation, artifact, test, benchmark, and profiling directories are added when those workstreams begin.

```text
transformer-inference-engine/
├─ docs/
├─ specs/
├─ artifacts/
├─ tools/
├─ implementations/
│  ├─ pytorch/
│  ├─ cpp/
│  └─ rust/
├─ tests/
├─ benchmarks/
└─ profiling/
```

### `docs/`
High-level documentation, roadmap notes, and implementation guidance.

### `specs/`
Repository-wide contracts such as:

- model config schema
- internal tensor naming
- weight artifact format
- cache layout
- test-vector format
- CLI contract

### `artifacts/`
Language-agnostic data produced and consumed by the project, such as:

- converted checkpoints
- tokenizer files
- test vectors
- prompt sets
- benchmark configs

### `tools/`
Utilities that glue the project together, including:

- checkpoint conversion
- test-vector export
- cross-language comparison
- benchmark orchestration

### `implementations/`
Language-specific engines. All three implementations should mirror the same major subsystem boundaries as closely as practical.

### `tests/`
Cross-language correctness, golden tests, regression tests, and property tests.

### `benchmarks/`
Benchmark suites, configs, and stored results.

### `profiling/`
Language-specific profiling notes and reports.

---

## Engine subsystems

The reusable engine is decomposed into the following subsystems.

### Config
Defines the engine-consumable model configuration. It is intentionally decoupled from Hugging Face config field names.

### Weights
Loads tensors from a shared artifact format into in-memory structures usable by the runtime.

### Tokenizer
Converts text to input IDs and back. In phase 0, tokenization parity is tracked separately from model-math correctness.

### Tensor and ops
Provides the basic numeric machinery used by higher-level modules.

### Modules
Implements reusable model blocks such as:

- attention
- MLP
- normalization
- residual composition

### Cache
Stores recurrent decode state. In phase 0 this is an append-only KV cache with canonical logical layout `[batch, kv_heads, seq, head_dim]`.

### Sampling
Implements greedy, temperature, top-k, and top-p sampling.

### Generation
Implements:

- prefill
- decode loop
- stop conditions
- output formatting

### Validation and benchmarking
Ensures that all implementations consume the same artifacts, run the same prompts, and can be compared under a shared policy.

---

## Phase roadmap

The roadmap is organized by engine concepts rather than by arbitrary model names.

### Phase 0 — GPT-2
Introduces:

- baseline decoder-only runtime
- learned absolute position embeddings
- LayerNorm
- GELU-style MLP
- append-only cache
- shared weight artifact format
- cross-language correctness harness

### Phase 1 — LLaMA-style support
Introduces:

- RoPE
- RMSNorm or equivalent modern norm flow
- gated MLP variant such as SwiGLU

### Phase 2 — MQA support
Introduces:

- decoupled Q-head and KV-head layout
- smaller KV cache design

### Phase 3 — GQA and rolling cache support
Introduces:

- grouped-query attention
- sliding-window or rolling cache policy

### Phase 4 — modern dense model support
Adds a family such as Qwen2.5 built mainly by composing features already introduced.

### Phase 5 — advanced hybrid support
Optional extension toward hybrid-attention families such as Qwen3.5-like models.

---

## Shared versus language-specific responsibilities

### Shared across all implementations
These should behave identically across PyTorch, C++, and Rust:

- model config semantics
- internal tensor names
- weight artifact layout
- cache semantics
- test-vector schema
- correctness tolerance policy
- benchmark input definitions
- CLI meaning at a behavioral level

### Allowed to differ by implementation
These may differ by language:

- internal tensor storage strategy
- memory management details
- threading model
- low-level numeric kernels
- profiling hooks
- exact package/build tooling

This boundary is important: the project compares implementations of the same engine contracts, not three unrelated programs.

---

## Explicit decisions for phase 0

### Decision 1: explicit `lm_head.weight` entry in the artifact index

Phase 0 stores `lm_head.weight` explicitly in the weight index, even when embeddings are tied.

Why this choice:

- it keeps the artifact schema simple and uniform across languages
- it avoids forcing every loader to implement alias semantics immediately
- it makes debugging and inspection easier during phase 0

Loaders may optimize by aliasing `lm_head.weight` to `tok_embeddings.weight` in memory when `tie_word_embeddings = true`, but that optimization is a loader concern, not an artifact-format requirement.

### Decision 2: tokenizer parity is separate from model-math correctness

Phase 0 correctness is split into two layers:

- model correctness from input IDs to logits and generated token IDs
- tokenizer correctness from text to input IDs and back

Why this choice:

- it removes a major debugging confound early on
- it lets C++ and Rust engines be validated before their tokenizer integration is perfect
- it keeps milestone 0.2 focused on inference math and runtime behavior

End-to-end text tests still matter, but they are not allowed to block model-math validation.

### Decision 3: weight artifact format is raw binary plus JSON index

Phase 0 uses:

- `weights.bin`
- `weights.index.json`
- `metadata.json`
- `config.json`

Why this choice:

- simple to implement in all three languages
- easy to inspect and debug
- friendly to later memory mapping
- avoids dependence on framework-specific serialization

---

## Success criterion for milestone 0.1

Milestone 0.1 is complete when:

- the docs in `docs/` clearly describe repository architecture and phase-0 scope
- the specs in `specs/` define stable formats and semantics
- there is no ambiguity about how GPT-2 artifacts should be converted and consumed
- phase 0.2 can begin with a PyTorch reference implementation without redefining repository contracts
