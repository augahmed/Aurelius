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
- `internal/tokenizer` defines the tokenizer boundary and ships with a simple byte tokenizer.
- `internal/model` defines shared model contracts and configuration.
- `internal/transformer` contains a deterministic toy transformer-style model that exercises the runtime path.
- `internal/sampler` provides greedy and temperature-based next-token selection.
- `internal/runtime` coordinates tokenization, model forward passes, and autoregressive generation.
- `internal/server` provides a minimal HTTP skeleton for future API work.
- `cmd/aurelius` exposes the prototype via a CLI.

## Package Layout

```text
aurelius/
  cmd/aurelius/        CLI entrypoint
  internal/tensor/     Basic tensor math primitives
  internal/tokenizer/  Tokenizer interfaces and prototype implementation
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
- [x] Expose a simple CLI
- [x] Add initial tests and architecture documentation
- [ ] Add real attention and KV cache plumbing
- [ ] Add pretrained weight loading
- [ ] Add streaming and benchmarking workflows

## Example CLI Usage

```bash
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
```

## Development Commands

```bash
gofmt -w ./cmd ./internal
go test ./...
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
```

## Notes

This repository does not attempt real model loading yet. The current focus is architectural clarity, stable tests, and a small inference path that future work can replace piece by piece.
