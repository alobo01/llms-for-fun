# Conceptual Guide to This Repository

This guide explains the repository's design intent, shared vocabulary, and
implementation direction. It is a conceptual guide, not a replacement for the
specifications, schemas, validation vectors, or implementation tests.

## Executive Summary

This repository defines the architecture, contracts, and roadmap for a
decoder-only language model inference engine intended to be implemented in
three languages:

- PyTorch;
- C++;
- Rust.

The project starts with GPT-2 as phase 0. GPT-2 is used to prove the engine
end to end with a CPU-first, correctness-first implementation before moving to
modern runtime features such as RoPE, RMSNorm, gated MLPs, MQA, GQA, and
sliding-window cache policies.

The repository is organized around shared engine contracts rather than around
three independent language ports. PyTorch, C++, and Rust implementations must
consume the same converted artifacts, use the same tensor names, obey the same
cache semantics, expose equivalent CLI behavior, and pass the same validation
vectors.

The current repository is milestone 0.1 material: it contains documentation and
specifications, not implementation code or model artifacts yet.

## Current Repository State

Current files:

```text
.
+-- roadmap.md
+-- repo_structure.md
+-- docs/
|   +-- architecture.md
|   +-- phase0-gpt2.md
|   +-- validation.md
+-- specs/
    +-- cache_layout.md
    +-- cli_contract.md
    +-- model_config_schema.json
    +-- tensor_naming.md
    +-- test_vector_format.md
    +-- weight_format.md
```

No implementation directories currently exist. The planned directories
`artifacts/`, `tools/`, `implementations/`, `tests/`, `benchmarks/`, and
`profiling/` are target structure, not current state.

## Source Documents

The project is described by these documents:

- `roadmap.md`: staged model-family and engine-feature roadmap.
- `repo_structure.md`: target repository organization and implementation work
  breakdown.
- `docs/architecture.md`: architecture principles and subsystem layout.
- `docs/phase0-gpt2.md`: GPT-2 phase-0 scope, deliverables, milestones, and
  success criteria.
- `docs/validation.md`: correctness layers, reference implementation, test
  classes, tolerances, and debugging workflow.
- `specs/model_config_schema.json`: engine-native model config schema.
- `specs/tensor_naming.md`: canonical internal tensor names.
- `specs/weight_format.md`: converted model artifact layout.
- `specs/cache_layout.md`: KV-cache logical layout and semantics.
- `specs/test_vector_format.md`: JSON and NPZ correctness-vector formats.
- `specs/cli_contract.md`: common CLI operations and structured outputs.

## Core Design Principle

Build one shared engine contract with per-language runtimes.

The engine should be organized around reusable concepts:

- specs;
- shared artifacts;
- language implementations;
- validation;
- benchmarks and profiling;
- roadmap-ready abstractions.

Adding a later model family should mostly mean adding a model-family adapter,
weight mapping, and a small number of new block variants. It should not mean
rewriting the runtime separately in each language.

## Design Principles

### Shared contracts first

The source of truth for repository-wide behavior lives in `specs/`.

Rules:

- `specs/` defines contracts.
- `artifacts/` contains data that follows those contracts.
- `implementations/` consume artifacts without redefining formats.
- `tools/` may generate or validate artifacts, but may not silently invent
  incompatible formats.

### Engine core separated from model-family support

Reusable engine subsystems include:

- config parsing;
- weight loading;
- tokenizer integration;
- tensor and buffer handling;
- numeric ops;
- transformer modules;
- KV-cache management;
- sampling;
- generation loop;
- validation and benchmarking.

Model-family-specific code should be limited to:

- checkpoint mapping into internal tensor names;
- layer and block structure;
- position handling strategy;
- norm and MLP variants;
- attention head layout details.

### CPU-first and correctness-first

Phase 0 is CPU-first. The initial target is correctness, clarity, and
extensibility, not peak performance. GPU support, quantization, speculative
decoding, paged attention, and distributed inference are later concerns.

### Prefill and decode are first-class paths

The runtime must explicitly distinguish:

- prefill: process a prompt sequence and build cache state;
- decode: process incremental generation steps using cached state.

This distinction exists from phase 0 because it affects correctness testing,
cache design, and benchmarking.

### Future-proof interfaces

GPT-2 is the first model family, but phase-0 interfaces should already leave
room for:

- learned absolute positions and RoPE;
- LayerNorm and RMSNorm;
- GELU MLPs and gated MLPs such as SwiGLU or GeGLU;
- MHA, MQA, and GQA;
- append-only and sliding-window cache policies.

The goal is future-proofing at the config and interface level, not
prematurely implementing every feature.

## Target Repository Structure

The full project should evolve toward:

```text
transformer-inference-engine/
+-- README.md
+-- roadmap.md
+-- docs/
|   +-- architecture.md
|   +-- phase0-gpt2.md
|   +-- validation.md
|   +-- benchmarks.md
|   +-- upgrade-guides/
|       +-- phase1-llama-style.md
|       +-- phase2-gemma.md
|       +-- phase3-mistral.md
|       +-- phase4-qwen2_5.md
+-- specs/
|   +-- engine_interfaces.md
|   +-- model_config_schema.json
|   +-- tensor_naming.md
|   +-- cache_layout.md
|   +-- weight_format.md
|   +-- sampling.md
|   +-- cli_contract.md
|   +-- test_vector_format.md
+-- artifacts/
|   +-- models/
|   +-- tokenizers/
|   +-- test_vectors/
|   +-- prompts/
+-- tools/
|   +-- convert_weights/
|   +-- export_test_vectors/
|   +-- compare_outputs/
|   +-- benchmark_runner/
|   +-- dev/
+-- implementations/
|   +-- pytorch/
|   +-- cpp/
|   +-- rust/
+-- tests/
|   +-- cross_language/
|   +-- golden/
|   +-- property/
|   +-- regression/
+-- benchmarks/
|   +-- suites/
|   +-- configs/
|   +-- results/
+-- profiling/
    +-- pytorch/
    +-- cpp/
    +-- rust/
```

Directory responsibilities:

- `docs/`: high-level documentation, roadmap notes, and implementation
  guidance.
- `specs/`: repository-wide contracts.
- `artifacts/`: converted checkpoints, tokenizer files, test vectors, prompts,
  and benchmark configs.
- `tools/`: conversion, test-vector export, comparison, benchmark, and dev
  utilities.
- `implementations/`: PyTorch, C++, and Rust engines.
- `tests/`: cross-language correctness, golden, property, and regression
  tests.
- `benchmarks/`: benchmark suites, configs, and stored results.
- `profiling/`: language-specific profiling notes and reports.

## Engine Subsystems

### Config

Defines the engine-consumable model configuration. It is intentionally
decoupled from upstream checkpoint field names.

### Weights

Loads tensors from the shared artifact format into runtime structures usable by
each language implementation.

### Tokenizer

Converts text to token IDs and token IDs back to text. In phase 0, tokenizer
parity is tracked separately from model-math correctness.

### Tensor and ops

Provides numeric machinery such as tensor storage, matmul, linear projection,
masking, softmax, normalization, and activations.

### Modules

Implements reusable transformer pieces:

- attention;
- MLP;
- normalization;
- residual composition;
- block execution.

### Cache

Stores recurrent decode state. Phase 0 uses append-only KV cache with logical
layout:

```text
[batch, kv_heads, seq, head_dim]
```

### Sampling

Implements:

- greedy decoding;
- temperature;
- top-k;
- top-p.

### Generation

Implements:

- prefill;
- decode loop;
- stop conditions;
- output formatting.

### Validation and benchmarking

Ensures all implementations consume the same artifacts, run the same prompts,
and report comparable results under shared policies.

## Roadmap

The roadmap is organized by engine concepts.

```text
GPT-2 -> LLaMA-style -> MQA -> GQA + sliding window -> modern dense -> hybrid
```

### Phase 0 - GPT-2

Goal: prove the baseline dense decoder runtime.

Introduces:

- token embeddings;
- learned absolute position embeddings;
- standard causal masked self-attention;
- LayerNorm;
- GELU-family MLP;
- logits;
- autoregressive generation;
- prefill/decode split;
- append-only KV cache;
- shared artifact format;
- cross-language validation harness.

### Phase 1 - LLaMA-style support

Goal: modernize the transformer core while preserving the dense decoder-only
runtime model.

Introduces:

- RoPE;
- RMSNorm or equivalent modern norm flow;
- gated MLP such as SwiGLU.

### Phase 2 - MQA support

Goal: decouple query heads from key/value heads.

Introduces:

- `num_attention_heads` versus `num_key_value_heads`;
- MQA broadcast rules;
- smaller KV-cache shape tests.

### Phase 3 - GQA and sliding-window support

Goal: support practical modern attention/cache behavior.

Introduces:

- grouped-query attention;
- windowed attention masks;
- rolling or fixed-size cache policy;
- attention-position bookkeeping for windowed decode.

### Phase 4 - Modern dense family support

Goal: support a realistic current dense model family, such as Qwen2.5-like
support, mostly by composing earlier features.

Likely additions:

- family-specific loader and config translation;
- long-context validation;
- QKV bias handling if not already covered.

### Phase 5 - Advanced hybrid support

Goal: support heterogeneous runtime patterns beyond standard dense
transformers.

Possible additions:

- hybrid token mixers;
- heterogeneous layer types;
- multiple cache or recurrent state abstractions;
- backend-specific optimizations.

## Phase 0 - GPT-2 Scope

Phase 0 establishes the first complete end-to-end inference engine
implementation. GPT-2 is chosen because it is simple enough to validate while
still exercising the core decoder-only runtime.

Required capabilities:

1. Load GPT-2 weights converted into the engine artifact format.
2. Accept token IDs as model input.
3. Run a full-sequence forward pass.
4. Run autoregressive decode using a KV cache.
5. Produce next-token logits.
6. Generate tokens with greedy, temperature, top-k, and top-p sampling.
7. Validate correctness against a PyTorch reference.
8. Support shared benchmark prompts and benchmark harness inputs.

Required implementation languages:

- PyTorch;
- C++;
- Rust.

Phase-0 non-goals:

- GPU kernels;
- quantization;
- speculative decoding;
- paged attention;
- sliding-window attention;
- MQA or GQA behavior beyond future-proof fields;
- non-GPT-2 families;
- distributed or parallel inference across devices.

## Phase 0 Deliverables

### Shared artifact pipeline

Converts GPT-2 reference checkpoints into the engine artifact format.

Expected output files:

- `config.json`;
- `metadata.json`;
- `weights.index.json`;
- `weights.bin`;
- tokenizer assets.

### Shared specs and docs

Stable documentation and specs for:

- architecture;
- weight format;
- cache layout;
- test-vector format;
- validation policy;
- CLI behavior.

### PyTorch reference engine

The PyTorch implementation is the first complete reference.

It should:

- load converted GPT-2 artifacts;
- run the forward pass;
- run cache-based decode;
- export golden vectors;
- serve as the correctness oracle.

### C++ engine

A C++ implementation consumes the same artifacts and matches PyTorch behavior
within tolerance.

### Rust engine

A Rust implementation consumes the same artifacts and matches PyTorch behavior
within tolerance.

### Cross-language validation harness

Compares:

- shapes;
- selected hidden states;
- logits;
- greedy next-token outputs;
- cache-equivalence behavior.

### Baseline benchmark harness

Measures at least:

- prefill latency;
- decode latency per token;
- end-to-end generation latency;
- small-batch throughput.

## Phase 0 Milestones

### Milestone 0.1 - Specs and artifact format

Deliver:

- `docs/architecture.md`;
- `docs/phase0-gpt2.md`;
- `docs/validation.md`;
- `specs/model_config_schema.json`;
- `specs/tensor_naming.md`;
- `specs/weight_format.md`;
- `specs/cache_layout.md`;
- `specs/test_vector_format.md`;
- `specs/cli_contract.md`.

### Milestone 0.2 - PyTorch reference correctness

Deliver:

- config loader;
- weight loader;
- GPT-2 forward pass;
- append-only KV cache;
- prefill and decode paths;
- greedy generation;
- golden test-vector exporter.

### Milestone 0.3 - C++ correctness engine

Deliver:

- same artifact consumption;
- same runtime behavior;
- matching smoke-test outputs within tolerance.

### Milestone 0.4 - Rust correctness engine

Deliver:

- same artifact consumption;
- same runtime behavior;
- matching smoke-test outputs within tolerance.

### Milestone 0.5 - Unified validation

Deliver:

- shared smoke tests;
- cache-equivalence tests;
- generation parity checks;
- cross-language comparison runner.

### Milestone 0.6 - Benchmark baseline

Deliver:

- benchmark prompt sets;
- benchmark configs;
- reporting scripts;
- first profiling summaries.

## Phase 0 Success Criteria

Phase 0 is successful when all three implementations can:

1. Load the same GPT-2 converted artifacts.
2. Produce matching final-position logits within tolerance.
3. Produce identical greedy next-token outputs.
4. Pass cache-equivalence tests.
5. Run the same benchmark prompts through a shared benchmark interface.

The main outcome is not only a working GPT-2 engine. It is a contract set that
makes later model-family support incremental.

## Model Config Contract

The model config is internal to the engine. It is not a Hugging Face config.

Required fields:

- `model_family`;
- `model_name`;
- `architecture`;
- `vocab_size`;
- `max_position_embeddings`;
- `hidden_size`;
- `num_hidden_layers`;
- `num_attention_heads`;
- `num_key_value_heads`;
- `intermediate_size`;
- `activation`;
- `norm_type`;
- `norm_epsilon`;
- `positional_encoding`;
- `mlp_type`;
- `attention_type`;
- `attention_bias`;
- `mlp_bias`;
- `tie_word_embeddings`;
- `bos_token_id`;
- `eos_token_id`;
- `dtype`.

Optional or nullable fields:

- `rope_theta`;
- `rope_scaling`;
- `initializer_range`.

Phase-0 GPT-2 example:

```json
{
  "model_family": "gpt2",
  "model_name": "gpt2",
  "architecture": "decoder_only",
  "vocab_size": 50257,
  "max_position_embeddings": 1024,
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "num_key_value_heads": 12,
  "intermediate_size": 3072,
  "activation": "gelu_new",
  "norm_type": "layer_norm",
  "norm_epsilon": 1e-5,
  "positional_encoding": "learned_absolute",
  "rope_theta": null,
  "rope_scaling": null,
  "mlp_type": "gelu_ffn",
  "attention_type": "mha",
  "attention_bias": true,
  "mlp_bias": true,
  "tie_word_embeddings": true,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "dtype": "float32"
}
```

Important semantic rules:

- phase 0 supports `architecture = "decoder_only"`;
- phase 0 artifact support requires `dtype = "float32"`;
- `float16` and `bfloat16` are reserved for later phases;
- `learned_absolute` positions require `rope_theta = null`;
- GPT-2 requires `attention_type = "mha"`;
- for GPT-2 and MHA, `num_key_value_heads == num_attention_heads`;
- `hidden_size` must be divisible by `num_attention_heads`;
- `head_dim = hidden_size / num_attention_heads`.

## Tensor Naming Contract

Converted artifacts use canonical internal tensor names. Runtime loaders must
not depend on upstream checkpoint names.

Naming conventions:

- dot-separated logical paths;
- zero-based transformer block index: `blocks.{i}`;
- standard parameter suffixes: `weight`, `bias`;
- names describe logical role, not framework implementation details.

Reserved top-level prefixes:

- `tok_embeddings`;
- `pos_embeddings`;
- `blocks`;
- `ln_f`;
- `lm_head`.

Required GPT-2 tensor names:

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

GPT-2 tensor shapes:

| Tensor | Shape |
| --- | --- |
| `tok_embeddings.weight` | `[vocab_size, hidden_size]` |
| `pos_embeddings.weight` | `[max_position_embeddings, hidden_size]` |
| `blocks.{i}.ln_1.weight` | `[hidden_size]` |
| `blocks.{i}.ln_1.bias` | `[hidden_size]` |
| `blocks.{i}.attn.qkv.weight` | `[3 * hidden_size, hidden_size]` |
| `blocks.{i}.attn.qkv.bias` | `[3 * hidden_size]` |
| `blocks.{i}.attn.out_proj.weight` | `[hidden_size, hidden_size]` |
| `blocks.{i}.attn.out_proj.bias` | `[hidden_size]` |
| `blocks.{i}.ln_2.weight` | `[hidden_size]` |
| `blocks.{i}.ln_2.bias` | `[hidden_size]` |
| `blocks.{i}.mlp.fc_in.weight` | `[intermediate_size, hidden_size]` |
| `blocks.{i}.mlp.fc_in.bias` | `[intermediate_size]` |
| `blocks.{i}.mlp.fc_out.weight` | `[hidden_size, intermediate_size]` |
| `blocks.{i}.mlp.fc_out.bias` | `[hidden_size]` |
| `ln_f.weight` | `[hidden_size]` |
| `ln_f.bias` | `[hidden_size]` |
| `lm_head.weight` | `[vocab_size, hidden_size]` |

GPT-2 stores attention input projection as fused QKV:

- `blocks.{i}.attn.qkv.weight`;
- `blocks.{i}.attn.qkv.bias`.

Later model families may use separate projection names such as:

- `blocks.{i}.attn.q_proj.weight`;
- `blocks.{i}.attn.k_proj.weight`;
- `blocks.{i}.attn.v_proj.weight`.

### Tied embeddings

`lm_head.weight` must appear explicitly in the artifact index even when
`tie_word_embeddings = true`.

Loaders may internally alias `lm_head.weight` to `tok_embeddings.weight`, but
the artifact format does not require alias semantics.

## Weight Artifact Contract

Phase 0 uses:

```text
<model_dir>/
  config.json
  metadata.json
  weights.index.json
  weights.bin
```

### `config.json`

Contains the internal engine model configuration.

### `metadata.json`

Required fields:

- `format_version`;
- `model_family`;
- `model_name`;
- `endianness`;
- `default_dtype`;
- `tensor_count`;
- `weight_file`;
- `index_file`.

Example:

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

### `weights.index.json`

Maps internal tensor names to location and shape metadata.

Required per-tensor fields:

- `dtype`;
- `shape`;
- `offset`;
- `nbytes`.

Optional per-tensor fields:

- `source_name`;
- `checksum`;
- `notes`.

Loaders must use explicit `offset` and `nbytes` fields. They must not rely on
JSON object key order.

### `weights.bin`

Raw binary tensor data.

Rules:

- little-endian;
- contiguous row-major storage;
- no compression in phase 0;
- no framework-specific headers;
- required phase-0 dtype support is `float32`.

### Artifact validation

A valid artifact must satisfy:

1. `config.json` conforms to `specs/model_config_schema.json`.
2. `metadata.json` names the correct weight and index files.
3. Every tensor in `weights.index.json` has a valid byte range within
   `weights.bin`.
4. Tensor byte size matches `shape * dtype_size`.
5. All required tensor names for the target model family are present.

### Conversion responsibilities

The converter must:

- read the source checkpoint;
- map source tensor names to internal tensor names;
- ensure tensors are contiguous in the chosen storage order;
- write `config.json`;
- write `metadata.json`;
- write `weights.index.json`;
- write `weights.bin`.

Runtime loading logic must not depend on upstream checkpoint names after
conversion.

## KV-Cache Contract

Canonical logical layout:

```text
keys:   [batch, kv_heads, seq, head_dim]
values: [batch, kv_heads, seq, head_dim]
```

This is a logical layout, not a mandatory physical memory layout.

Phase-0 GPT-2 cache:

- `num_key_value_heads == num_attention_heads`;
- `head_dim = hidden_size / num_attention_heads`;
- cache policy is `append_only`;
- dtype defaults to `float32`.

For each layer `i`, the cache contains:

```text
K_i: [batch, kv_heads, seq, head_dim]
V_i: [batch, kv_heads, seq, head_dim]
```

Required operations:

1. Initialize empty per-layer cache.
2. Append keys and values during prefill.
3. Append keys and values during decode.
4. Read the visible cache span.
5. Report current sequence length.
6. Clear cache state.

Phase 0 visible span is the entire accumulated sequence.

Cache semantics:

- token order must be preserved exactly;
- decode appends after the last prefill position;
- batch, KV-head, sequence, and head dimensions must remain consistent within
  a layer;
- implementations may optimize physical memory layout if logical semantics are
  preserved.

Cache equivalence is mandatory:

```text
full recomputation on P + t
==
prefill on P followed by one-step decode on t using cache
```

The final-position logits must match within tolerance.

## Test-Vector Contract

Test vectors are produced by the PyTorch reference and consumed by all
implementations.

Phase 0 uses:

- JSON for smoke and regression cases;
- NPZ for larger tensor dumps and intermediate values.

Recommended path:

```text
artifacts/test_vectors/<family>/<model_name>/
```

For GPT-2:

```text
artifacts/test_vectors/gpt2/gpt2/
```

Example contents:

```text
smoke_hello.json
smoke_paragraph.json
cache_equivalence_case_01.json
blockwise_case_01.npz
blockwise_case_01.json
final_logits_case_01.npz
final_logits_case_01.json
```

### JSON smoke tests

Required fields:

- `name`;
- `model_family`;
- `model_name`;
- `input_ids`;
- `dtype`;
- `tolerance`.

Optional fields:

- `prompt_text`;
- `expected_next_token_greedy`;
- `expected_generated_ids`;
- `logits_last_token_topk`;
- `sampling`;
- `notes`.

Example:

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

### NPZ tensor tests

Every NPZ file must have a JSON sidecar with the same basename.

Sidecar required fields:

- `name`;
- `model_family`;
- `model_name`;
- `dtype`;
- `tolerance`;
- `arrays`.

Recommended GPT-2 array names:

- `input_ids`;
- `position_ids`;
- `hidden_after_embeddings`;
- `hidden_after_block_0`;
- `hidden_after_block_N`;
- `final_hidden`;
- `final_logits`;
- `prefill_logits`;
- `decode_logits`;
- `greedy_generated_ids`.

### Tokenization policy

For model correctness vectors:

- `input_ids` are authoritative;
- `prompt_text` is optional metadata;
- if `prompt_text` and `input_ids` disagree, the vector is malformed.

### Tolerances

Every test vector must carry explicit tolerance settings:

```json
{
  "tolerance": {
    "atol": 1e-4,
    "rtol": 1e-4
  }
}
```

Phase-0 defaults:

- hidden states: `atol = 1e-4`, `rtol = 1e-4`;
- logits: `atol = 1e-4`, `rtol = 1e-4`;
- greedy next token: exact integer match.

### Required test classes

The PyTorch reference should export:

1. Smoke next-token test.
2. Multi-token greedy generation test.
3. Cache-equivalence test.
4. Intermediate tensor regression test.
5. Final logits regression test.

## Validation Strategy

Validation ensures PyTorch, C++, and Rust implement the same engine semantics
over the same artifacts.

Validation layers:

- model-math validation;
- tokenizer validation.

Model-math validation covers:

- input IDs;
- embeddings;
- transformer blocks;
- final hidden state;
- logits;
- decode behavior.

Tokenizer validation covers:

- prompt text to input IDs;
- output IDs back to text.

Tokenizer validation is important but separate. Phase-0 engine math is
validated from token IDs to logits and generated token IDs.

### Reference implementation

The PyTorch engine is the correctness reference for phase 0.

Responsibilities:

- load converted GPT-2 artifacts;
- run forward pass;
- run prefill and decode;
- export golden vectors;
- provide expected outputs for cross-language tests.

C++ and Rust validate against the same artifacts and expected outputs.

### Required correctness checks

Every implementation must support:

1. Shape correctness.
2. Dtype correctness.
3. Logits agreement.
4. Greedy next-token agreement.
5. Multi-step generation agreement.
6. Cache equivalence.

### Validation workflow

Recommended workflow:

1. Convert GPT-2 reference weights into the engine artifact format.
2. Run the PyTorch reference on selected prompts or input IDs.
3. Export JSON and NPZ golden vectors.
4. Run the same cases in C++ and Rust.
5. Compare outputs using a shared comparison tool.
6. Record regressions under `tests/regression/`.

### Debugging order

When a cross-language mismatch appears, debug in this order:

1. Confirm config fields and tensor shapes.
2. Confirm weight-name mapping and tensor loading.
3. Confirm embedding outputs.
4. Confirm block-by-block hidden-state agreement.
5. Confirm masking and attention score path.
6. Confirm cache append semantics and position handling.
7. Confirm final logits and sampling behavior.

Benchmarking is not validation. Performance claims are meaningful only after
correctness checks pass.

## CLI Contract

The CLI contract defines equivalent user-facing operations across
implementations.

Required operations:

```bash
engine run --model <model_dir> --prompt "Hello"
engine run --model <model_dir> --input-ids 15496,995
engine generate --model <model_dir> --prompt "Hello" --max-new-tokens 16
engine generate --model <model_dir> --input-ids 15496,995 --max-new-tokens 16
engine test --model <model_dir> --test-vector <path>
engine bench --model <model_dir> --suite <suite_config>
```

Correctness workflows should prefer `--input-ids`.

Recommended global arguments:

- `--model <model_dir>`;
- `--dtype <dtype>`;
- `--device <device>`;
- `--output <path>`;
- `--output-format <json|text>`;
- `--verbose`.

Phase 0 only requires CPU support. `--device` may exist for forward
compatibility.

### `run`

Purpose: execute a forward pass and return final-position information.

Required input:

- one of `--prompt` or `--input-ids`.

Required output capability:

- `input_ids`;
- `next_token_greedy`;
- `top_logits`.

### `generate`

Purpose: run autoregressive generation.

Required input:

- one of `--prompt` or `--input-ids`;
- `--max-new-tokens`.

Optional sampling inputs:

- `--temperature`;
- `--top-k`;
- `--top-p`;
- `--seed`.

Required output capability:

- `input_ids`;
- `generated_ids`;
- `num_generated_tokens`;
- `stopped_reason`.

### `test`

Purpose: load a test vector, execute the relevant model operation, compare
outputs using the vector tolerance, and emit pass/fail results.

### `bench`

Purpose: run a benchmark suite.

Required benchmark measurements:

- prefill latency;
- decode latency per token;
- end-to-end generation latency;
- throughput for one or more prompt cases.

### Output policy

All automation-facing commands must support `--output-format json`.

Human-readable text output is optional.

### Exit code policy

Required:

- `0` for success;
- nonzero for failure.

Recommended meanings:

- `1`: generic runtime or loading failure;
- `2`: CLI argument or usage error;
- `3`: validation or test failure.

JSON error example:

```json
{
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "Missing weights.index.json in model directory"
  }
}
```

The `--model` argument must point to the converted model artifact directory,
not an upstream Hugging Face cache directory.

Minimum phase-0 CLI compliance:

- accepts `--model`;
- supports `run`, `generate`, `test`, and `bench`;
- supports `--input-ids`;
- supports JSON output;
- returns nonzero on failure;
- consumes the shared GPT-2 converted artifact format.

## Implementation Layouts

### PyTorch

Target layout:

```text
src/engine_pt/
  config/
    model_config.py
  weights/
    loader.py
    mmap_loader.py
  tokenizer/
    gpt2_tokenizer.py
  tensor/
    dtypes.py
  ops/
    linear.py
    softmax.py
    mask.py
    norms.py
    activations.py
  modules/
    attention.py
    mlp.py
    block.py
  cache/
    kv_cache.py
  sampling/
    greedy.py
    topk.py
    topp.py
    temperature.py
  generation/
    prefill.py
    decode.py
    generator.py
  models/
    gpt2/
      config_adapter.py
      weight_mapper.py
      model.py
  cli/
    main.py
  tests/
```

PyTorch is the reference implementation, artifact/test-vector generator, and
oracle for correctness.

### C++

Target layout:

```text
include/engine_cpp/
src/
  config/
  weights/
  tokenizer/
  tensor/
  ops/
  modules/
  cache/
  sampling/
  generation/
  models/
    gpt2/
  cli/
  tests/
```

C++ should consume the same artifacts, preserve the same cache semantics, and
match PyTorch outputs within tolerance before optimization work.

### Rust

Target layout:

```text
crates/
  engine-core/
    src/
      config.rs
      cache.rs
      sampling.rs
      traits.rs
  engine-ops/
    src/
      linear.rs
      softmax.rs
      norms.rs
      activations.rs
  engine-models/
    src/
      gpt2/
        config.rs
        mapper.rs
        attention.rs
        mlp.rs
        block.rs
        model.rs
  engine-cli/
  engine-tests/
tests/
```

Recommended Rust crate responsibilities:

- `engine-core`: config, traits, cache abstractions, sampler abstractions;
- `engine-ops`: numeric ops and kernels;
- `engine-models`: GPT-2 and later family support;
- `engine-cli`: command-line runner;
- `engine-tests`: shared test harness glue.

## Shared Versus Language-Specific Responsibilities

Shared across implementations:

- model artifact format;
- model config semantics;
- tensor naming;
- prompts;
- test-vector schema;
- expected outputs;
- benchmark suite definitions;
- CLI semantics;
- correctness tolerance policy;
- cache semantics.

Allowed to differ by language:

- internal tensor storage;
- memory management;
- threading model;
- numeric kernel implementation;
- profiling hooks;
- package and build tooling.

The project compares implementations of the same engine contracts, not three
unrelated programs.

## Upgrade-Ready Abstractions

Phase 0 should already separate:

- `ModelConfig`;
- `WeightLoader`;
- `AttentionKernel`;
- `PositionalEncoding`;
- `NormBlock`;
- `MLPBlock`;
- `CachePolicy`;
- `ModelFamilyAdapter`.

### Positional encoding

Phase 0:

- learned absolute positions.

Later:

- RoPE;
- RoPE scaling variants.

### Norm strategy

Phase 0:

- LayerNorm.

Later:

- RMSNorm.

### FFN strategy

Phase 0:

- GELU-family MLP.

Later:

- SwiGLU;
- GeGLU.

### Attention head layout

Phase 0:

- MHA with equal query heads and KV heads.

Later:

- MQA;
- GQA.

### Cache policy

Phase 0:

- append-only.

Later:

- sliding-window;
- rolling cache;
- paged or segmented cache.

## Engineering Rules

Do not couple model family names to engine core.

Avoid:

```text
Gpt2Tensor
Gpt2Cache
Gpt2Sampler
```

Prefer:

```text
Tensor
KVCache
Sampler
Gpt2Model
```

Rules:

- Separate checkpoint naming from internal naming.
- Treat upstream names as conversion input only.
- Keep prefill and decode as separate paths from phase 0.
- Build cache-equivalence tests early.
- Make benchmark inputs explicit and versioned.
- Treat PyTorch as the reference and tooling-heavy implementation.
- Treat C++ and Rust as equivalent implementations of the same contracts.

## Near-Term Backlog

The next practical steps are:

1. Keep the current docs and specs as milestone 0.1 contract material.
2. Add a top-level `README.md` that points to the key docs.
3. Add target directories only when each workstream begins.
4. Implement a GPT-2 weight converter.
5. Implement the PyTorch reference engine.
6. Export JSON and NPZ golden vectors.
7. Build the shared comparison tool.
8. Implement C++ and Rust engines against the same artifacts.
9. Add benchmark runner and profiling summaries after correctness is stable.

Useful future docs/specs:

- `specs/engine_interfaces.md`;
- `specs/sampling.md`;
- `docs/benchmarks.md`;
- `docs/upgrade-guides/phase1-llama-style.md`;
- `docs/upgrade-guides/phase2-gemma.md`;
- `docs/upgrade-guides/phase3-mistral.md`;
- `docs/upgrade-guides/phase4-qwen2_5.md`.

## Glossary

Append-only cache: KV-cache policy where newly computed positions are appended
and the full accumulated sequence remains visible.

Cache equivalence: validation that full recomputation on `P + t` matches
prefill on `P` followed by one-step decode on `t`.

Decode: incremental generation path that uses cached state.

GQA: grouped-query attention, where many query heads share fewer KV heads.

MHA: multi-head attention, where query heads and KV heads are equal.

MQA: multi-query attention, where many query heads share one or very few KV
heads.

Prefill: prompt processing path that initializes cache state.

RoPE: rotary positional embedding.

SwiGLU and GeGLU: gated feed-forward network variants used by newer
transformer families.

## Key Project Invariants

The following invariants should hold unless a later spec explicitly changes
them:

1. Shared specs define behavior; implementations consume them.
2. Converted artifacts use internal tensor names, not upstream names.
3. `weights.index.json` byte offsets are authoritative.
4. Phase-0 artifact support is `float32`.
5. GPT-2 and MHA require `num_key_value_heads == num_attention_heads`.
6. KV-cache logical layout is `[batch, kv_heads, seq, head_dim]`.
7. Model-math validation is separate from tokenizer validation.
8. Cache equivalence is mandatory.
9. PyTorch is the phase-0 correctness reference.
10. C++ and Rust must match the same artifacts and test vectors.

## Integrated Files

This guide integrates the current content and decisions from:

- `roadmap.md`;
- `repo_structure.md`;
- `docs/architecture.md`;
- `docs/phase0-gpt2.md`;
- `docs/validation.md`;
- `specs/cache_layout.md`;
- `specs/cli_contract.md`;
- `specs/model_config_schema.json`;
- `specs/tensor_naming.md`;
- `specs/test_vector_format.md`;
- `specs/weight_format.md`.
