# CLI Contract Specification

## Purpose

This document defines the minimal command-line interface contract shared by the PyTorch, C++, and Rust implementations of the inference engine.

The goal is not to force identical implementation details, but to ensure that all three runtimes expose equivalent user-facing operations for correctness testing, debugging, and benchmarking.

---

## Design principles

The CLI contract follows these rules:

- commands should map to engine concepts rather than language-specific tooling
- all implementations should accept the same model artifact format
- all implementations should support token-ID-driven correctness workflows
- JSON output should be available for machine-readable validation
- benchmark invocation should be consistent enough for shared automation

---

## Required commands

Phase 0 requires the following commands or exact-equivalent subcommands:

```bash
engine run --model <model_dir> --prompt "Hello"
engine run --model <model_dir> --input-ids 15496,995
engine generate --model <model_dir> --prompt "Hello" --max-new-tokens 16
engine generate --model <model_dir> --input-ids 15496,995 --max-new-tokens 16
engine test --model <model_dir> --test-vector <path>
engine bench --model <model_dir> --suite <suite_config>
```

Implementations may expose additional options, but these operations form the minimum contract.

---

## Phase 0 decision

The CLI must support both `--prompt` and `--input-ids`, but correctness workflows should prefer `--input-ids`.

### Justification

This matches the broader phase 0 decision to separate tokenizer correctness from model-math correctness. It also makes cross-language validation much easier because token ID inputs eliminate tokenizer drift as a confounder.

---

## Global arguments

The following global arguments are recommended across commands:

- `--model <model_dir>`
- `--dtype <dtype>`
- `--device <device>`
- `--output <path>`
- `--output-format <json|text>`
- `--verbose`

### Phase 0 decision

`--device` may exist for forward compatibility, but phase 0 implementations are only required to support CPU.

### Justification

Keeping `--device` in the CLI now avoids later command churn when GPU support is introduced.

---

## `run` command

### Purpose

Execute a forward pass and return information about the final position.

### Required inputs

One of:

- `--prompt <text>`
- `--input-ids <comma-separated-token-ids>`

### Required behavior

- load the model from the provided artifact directory
- tokenize prompt text if `--prompt` is used
- run a forward pass on the full input sequence
- return final-position information

### Required outputs

At minimum, `run` must be able to emit:

- `input_ids`
- `next_token_greedy`
- `top_logits`

### Recommended JSON output

```json
{
  "model_family": "gpt2",
  "model_name": "gpt2",
  "input_ids": [15496, 995],
  "next_token_greedy": 0,
  "top_logits": [
    {"token_id": 0, "logit": 11.23},
    {"token_id": 13, "logit": 10.91}
  ]
}
```

---

## `generate` command

### Purpose

Run autoregressive generation.

### Required inputs

One of:

- `--prompt <text>`
- `--input-ids <comma-separated-token-ids>`

And:

- `--max-new-tokens <int>`

### Optional sampling inputs

- `--temperature <float>`
- `--top-k <int>`
- `--top-p <float>`
- `--seed <int>`

### Required behavior

- perform prefill on the prompt
- perform iterative decode using the KV cache
- stop at EOS or after `max_new_tokens`

### Required outputs

At minimum, `generate` must be able to emit:

- `input_ids`
- `generated_ids`
- `num_generated_tokens`

### Recommended JSON output

```json
{
  "model_family": "gpt2",
  "model_name": "gpt2",
  "input_ids": [15496, 995],
  "generated_ids": [15496, 995, 0, 13],
  "num_generated_tokens": 2,
  "stopped_reason": "max_new_tokens"
}
```

---

## `test` command

### Purpose

Run a correctness check using a shared test vector.

### Required inputs

- `--model <model_dir>`
- `--test-vector <path>`

### Required behavior

- load the test vector
- execute the relevant model operation
- compare outputs against expected values using the vector’s tolerance policy
- emit machine-readable pass/fail results

### Required JSON output

```json
{
  "test_name": "smoke_hello",
  "passed": true,
  "checks": [
    {"name": "shape", "passed": true},
    {"name": "next_token_greedy", "passed": true},
    {"name": "top_logits", "passed": true}
  ]
}
```

---

## `bench` command

### Purpose

Run a benchmark suite with standardized prompts and settings.

### Required inputs

- `--model <model_dir>`
- `--suite <suite_config>`

### Required behavior

A phase 0 benchmark suite should be able to measure at least:

- prefill latency
- decode latency per token
- end-to-end generation latency
- throughput for one or more prompt cases

### Required output fields

Recommended JSON output should include:

- suite name
- model name
- hardware summary if available
- metrics dictionary

Example:

```json
{
  "suite": "gpt2_smoke",
  "model_family": "gpt2",
  "model_name": "gpt2",
  "metrics": {
    "prefill_latency_ms": 3.21,
    "decode_latency_ms_per_token": 0.84,
    "end_to_end_latency_ms": 15.02,
    "tokens_per_second": 1190.5
  }
}
```

---

## Output format policy

### Phase 0 decision

All commands should support `--output-format json`. Human-readable text output is optional, but JSON output is mandatory for automation-facing paths.

### Justification

Cross-language correctness and benchmarking workflows depend on structured output. JSON is the simplest interoperable choice across PyTorch, C++, and Rust.

---

## Exit code policy

The CLI must follow this exit code convention:

- `0` for successful execution
- nonzero for failure

Recommended meanings:

- `1` generic runtime or loading failure
- `2` CLI argument or usage error
- `3` validation or test failure

This convention is recommended rather than mandatory, but all implementations must at least distinguish success from failure.

---

## Error output policy

When execution fails, the CLI should emit structured JSON if `--output-format json` is selected.

Example:

```json
{
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "Missing weights.index.json in model directory"
  }
}
```

---

## Artifact path policy

The `--model` argument must point to the converted model artifact directory, not to an upstream Hugging Face cache directory.

For phase 0 GPT-2, that means a directory containing at least:

- `config.json`
- `metadata.json`
- `weights.index.json`
- `weights.bin`

This ensures all implementations consume the same canonical artifacts.

---

## Forward-compatibility guidance

The CLI should be designed so that later phases can add support for:

- family selection or automatic detection from config
- device backends
- quantized artifacts
- cache policy selection
- profiling modes

Without changing the phase 0 command names.

---

## Minimum compliance checklist

A phase 0 implementation is CLI-compliant if it:

- accepts `--model`
- supports `run`, `generate`, `test`, and `bench`
- supports `--input-ids`
- supports JSON output
- returns nonzero on failure
- can consume the shared GPT-2 converted artifact format

This is the minimum standard required for cross-language validation and benchmarking automation.
