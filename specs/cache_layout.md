# KV Cache Layout

## Purpose

This document defines the canonical logical KV-cache layout and cache semantics for the repository.

Phase 0 uses GPT-2, but the cache contract is designed so later support for MQA, GQA, and rolling cache policies can reuse the same conceptual model.

---

## Phase-0 decision

The canonical logical cache layout is:

- keys: `[batch, kv_heads, seq, head_dim]`
- values: `[batch, kv_heads, seq, head_dim]`

This is the chosen layout because it generalizes directly from GPT-2 multi-head attention to later MQA and GQA models.

Although GPT-2 has `num_key_value_heads == num_attention_heads`, phase 0 uses the name `kv_heads` in cache layouts now so later phases do not require renaming the cache contract.

---

## Terminology

### Prefill

The operation that processes an input prompt sequence and initializes cache state for all processed positions.

### Decode

The incremental generation path that processes one or more newly generated positions using previously cached state.

### Cache policy

The rule that governs how recurrent state is stored and updated.

Phase 0 uses `append_only`.

---

## GPT-2 phase-0 cache

For GPT-2 in phase 0:

- `num_key_value_heads == num_attention_heads`
- `head_dim = hidden_size / num_attention_heads`
- cache policy = `append_only`

For each transformer layer `i`, the cache contains:

- `K_i`: `[batch, kv_heads, seq, head_dim]`
- `V_i`: `[batch, kv_heads, seq, head_dim]`

---

## Required cache operations

Every implementation must support these logical operations.

### 1. Initialize empty cache

Create empty per-layer key and value tensors with zero sequence length.

### 2. Append during prefill

Given one or more prompt positions, append the newly computed keys and values to each layer cache.

### 3. Append during decode

Given one or more incremental decode positions, append the newly computed keys and values to each layer cache.

### 4. Read visible cache span

Expose the current sequence span that should be attended to during decode.

In phase 0 this is the entire accumulated sequence because the cache policy is append-only.

---

## Semantics

## Sequence order

The cache must preserve token order exactly.

If prompt positions correspond to time steps `0..T-1`, then after prefill the cache sequence dimension must reflect that same ordering.

If the next generated token is at time step `T`, appending during decode must place it after the last prefilling position.

## Batch consistency

All cached tensors within a layer must agree on batch dimension, KV-head dimension, and head dimension.

Phase 0 may assume simple batch handling, but the logical cache layout must still include the batch axis explicitly.

## Dtype consistency

The cache dtype should match the active runtime dtype for the model unless the implementation explicitly documents an internal cache storage choice.

Phase 0 artifact support is float32, so float32 cache behavior is the default expectation.

---

## Cache equivalence requirement

This repository treats cache equivalence as mandatory.

For a prompt `P` and next token `t`, the following two computations must produce matching final-position logits within tolerance:

1. full recomputation on `P + t`
2. prefill on `P`, then one-step decode on `t` using the cache

This test must pass for every implementation.

Why this matters:

- it validates cache append semantics
- it validates masking and position handling during decode
- it is the simplest strong test of autoregressive correctness

---

## Memory-layout freedom versus logical-layout contract

The canonical layout in this document is a logical layout, not a mandatory in-memory byte layout.

Implementations may choose different internal physical memory arrangements for performance reasons, as long as they preserve the same semantics and can reason about tensors in the canonical logical form.

This distinction is important:

- the spec defines correctness and interoperability
- implementations remain free to optimize storage internally

---

## Why append-only is chosen for phase 0

Phase 0 uses `append_only` because it is the simplest correct cache policy for GPT-2 and directly supports the main learning objectives:

- separating prefill from decode
- validating autoregressive correctness
- benchmarking decode latency

Sliding-window or rolling-cache behavior belongs to later milestones because it changes both cache semantics and attention visibility rules.

---

## Reserved extension path

The cache contract is intentionally designed so later phases can add:

- `append_only`
- `sliding_window(window_size)`
- paged or segmented cache policies

without changing the canonical logical role of:

- batch
- kv_heads
- seq
- head_dim

---

## Implementation guidance

Recommended internal cache API concepts include:

- `CacheConfig`
- `LayerCache`
- `KVCache`
- `append(layer_idx, new_k, new_v)`
- `current_length()`
- `clear()`

Exact type definitions may differ across PyTorch, C++, and Rust, but the semantic operations should remain aligned.

---

## Explicit decision: no sequence-major canonical contract

A possible alternative would have been sequence-major logical layout such as `[seq, batch, kv_heads, head_dim]`.

That is not the chosen canonical layout.

The chosen canonical layout remains `[batch, kv_heads, seq, head_dim]` because:

- it aligns naturally with per-request cache reasoning
- it is easy to describe for batched decode
- it generalizes clearly to MQA and GQA
- it keeps the batch axis first, which is convenient for shared validation and debugging

---

## Phase-0 validation checklist

A cache implementation is phase-0 compliant when:

1. it stores per-layer keys and values
2. it supports append-only growth in sequence order
3. it distinguishes prefill from decode behavior
4. it passes cache equivalence tests against full recomputation
5. it uses the canonical logical dimensions `batch`, `kv_heads`, `seq`, and `head_dim`
