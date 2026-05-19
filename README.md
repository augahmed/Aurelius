# Aurelius

Aurelius is a Go-based transformer inference runtime built from scratch for small, correct, extensible decoding experiments.

## Status

Early prototype.

## Long-Term Goals

- Build a clean transformer inference core in Go.
- Support autoregressive decoding with reusable KV cache abstractions.
- Add tokenizer, model loader, batching, streaming, and benchmarking layers incrementally.
- Create a foundation that can grow toward GPT-2 and LLaMA-style inference paths without rewriting the architecture.

## Architecture Overview

The current prototype is intentionally small:

- `internal/tensor` provides basic CPU tensor math for correctness-first experimentation.
- `internal/tokenizer` defines the tokenizer boundary and ships with both a simple byte tokenizer and a GPT-2 style BPE tokenizer loader.
- `internal/model` defines shared model contracts and configuration.
- `internal/arithmetic` generates synthetic arithmetic datasets and builds training sequences.
- `internal/gpt2` loads GPT-2 model config metadata from local `config.json` files.
- `internal/mathlm` contains a small trainable autoregressive MLP language model for student-scale arithmetic experiments.
- `internal/transformer` contains a deterministic toy transformer-style model that exercises the runtime path.
- `internal/sampler` provides greedy and temperature-based next-token selection.
- `internal/runtime` coordinates tokenization, model forward passes, and autoregressive generation, including optional KV-cached decoding when a model supports it.
- `internal/server` serves a lightweight local web app and JSON API for browser-based interaction.
- `cmd/aurelius` exposes the prototype via a CLI.

## Package Layout

```text
aurelius/
  cmd/aurelius/        CLI entrypoint
  internal/tensor/     Basic tensor math primitives
  internal/tokenizer/  Tokenizer interfaces and prototype implementation
  internal/gpt2/       GPT-2 asset config loader
  internal/model/      Shared model interfaces and config
  internal/transformer/Tiny transformer-style prototype model
  internal/sampler/    Token sampling strategies
  internal/runtime/    Generation engine
  internal/server/     Minimal HTTP server skeleton
  benchmarks/          Benchmark harnesses and notes
  docs/                Architecture and design notes
```

## First Milestone

- [x] Initialize Go module and repository layout
- [x] Define tokenizer, sampler, and model interfaces
- [x] Implement correctness-first tensor operations
- [x] Build a deterministic toy transformer inference path
- [x] Add an autoregressive runtime loop
- [x] Add optional KV-cached autoregressive generation in the runtime
- [x] Expose a simple CLI
- [x] Add a local chat-style web UI served by Go
- [x] Add initial tests and architecture documentation
- [ ] Add pretrained weight loading
- [ ] Add student-scale training workflow from scratch
- [ ] Add streaming and benchmarking workflows

## Example CLI Usage

```bash
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius generate -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius generate-gpt2 -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -max-tokens 1
go run ./cmd/aurelius emit-gpt2-observation -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -top-k 5
go run ./cmd/aurelius inspect-gpt2-next -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -top-k 5
go run ./cmd/aurelius validate-gpt2 -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -fixture /path/to/reference.json
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic
go run ./cmd/aurelius train-math -data-dir ./data/arithmetic -checkpoint ./artifacts/mathlm.json
go run ./cmd/aurelius eval-math -checkpoint ./artifacts/mathlm.json -data ./data/arithmetic/val.jsonl
go run ./cmd/aurelius generate-math -checkpoint ./artifacts/mathlm.json -prompt "12 + 7 = "
go run ./cmd/aurelius serve
go run ./cmd/aurelius serve -backend gpt2
go run ./cmd/aurelius serve -addr localhost:8080
go run ./cmd/aurelius tokenize -vocab /path/to/vocab.json -merges /path/to/merges.txt -text "hello world"
go run ./cmd/aurelius inspect-model -model-config /path/to/config.json -vocab /path/to/vocab.json -merges /path/to/merges.txt
```

## Web UI

Start the local server:

```bash
go run ./cmd/aurelius serve
```

Then open `http://localhost:8080`.

`serve` now supports `-backend auto|toy|gpt2`. In `auto` mode, Aurelius uses the GPT-2 backend when a complete checkpoint exists under `artifacts/gpt2/`; otherwise it falls back to the toy model.

The web UI provides:

- a polished chat-style interface
- prompt entry
- max token control
- temperature control
- top-k control
- cache usage toggle
- browser-side history persisted in `localStorage`
- basic markdown rendering for assistant responses

When `serve` is using the GPT-2 backend, the web path now applies conservative request limits for responsiveness: a short assistant preamble, bounded temperature and top-k sampling, modest default reply length, capped generation length, trimmed conversation history, and cache-aware incremental decoding.

## Arithmetic Training

Aurelius now includes a student-scale from-scratch training path for arithmetic. It uses:

- synthetic arithmetic JSONL datasets
- the existing byte tokenizer for a minimal training-first path
- a small fixed-context autoregressive MLP language model
- JSON checkpoints for save/resume
- exact-match evaluation on held-out arithmetic prompts

This path is intentionally small and educational. It is not a frontier LLM training stack and it does not replace the existing GPT-2 inference path.

## Development Commands

```bash
gofmt -w ./cmd ./internal
go test ./...
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius serve
```

## Notes

This repository does not attempt real model loading yet. The current focus is architectural clarity, stable tests, and a small inference path that future work can replace piece by piece.

Cache-aware generation is optional per model. `runtime.Engine` detects models that implement the cache-capable extension and uses incremental decoding only for those models; all other models continue to use the uncached full-sequence path.

The local web UI uses the same runtime engine and generation options as the CLI. Chat history is stored in the browser for now rather than persisted server-side.

The `generate-gpt2` path now runs a real non-cached GPT-2 style forward pass from loaded safetensors weights, while `inspect-gpt2-next` and `validate-gpt2` support parity validation. The default CLI and web UI still use the toy transformer until the GPT-2 path has broader real-checkpoint validation and feature parity.
