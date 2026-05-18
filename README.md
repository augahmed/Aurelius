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
- `internal/gpt2` loads GPT-2 model config metadata from local `config.json` files.
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
- [ ] Add streaming and benchmarking workflows

## Example CLI Usage

```bash
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius generate -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius serve
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

The web UI provides:

- a polished chat-style interface
- prompt entry
- max token control
- cache usage toggle
- browser-side history persisted in `localStorage`
- basic markdown rendering for assistant responses

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

The GPT-2 tokenizer and config loaders are asset-loading steps only. They do not yet replace the toy transformer with pretrained inference.
