# Aurelius

Aurelius is a Go implementation of a small transformer inference and training stack for math-focused language model experiments. It includes a byte-tokenized causal transformer, synthetic math curriculum generation, exact-match evaluation, checkpoint export tooling, and a local web interface backed by a math router.

## Status

Active research prototype. The codebase is designed for correctness, inspection, and reproducible small-model experiments rather than large-scale distributed training.

## Capabilities

- Train small autoregressive MLP and transformer language models from JSONL math datasets.
- Generate arithmetic, multiplication, and polynomial derivative curricula with metadata for grouped evaluation.
- Evaluate checkpoints by exact-match accuracy across operation, level, prompt template, answer length, and other tags.
- Serve a browser chat interface through the Go HTTP server.
- Route natural-language math prompts through specialist checkpoints with deterministic fallback for supported expressions.
- Export inference-only checkpoint artifacts by stripping optimizer state from trainer checkpoints.
- Load GPT-2 style tokenizer and weight assets for parity-oriented inference experiments.

## Quickstart

Clone the repository and run the test suite:

```bash
git clone https://github.com/augahmed/Aurelius.git
cd Aurelius
go test ./...
```

Start the local web server:

```bash
go run ./cmd/aurelius serve
```

Open `http://localhost:8080`. Without checkpoint flags, the server uses the built-in toy backend so the web path can be tested immediately.

## Math Router

The highest-accuracy math path is the `math-router` backend. It normalizes supported user prompts into direct model prompts, asks the trained specialist checkpoint first, and falls back to deterministic math evaluation when the model answer does not exactly match the computed result.

Place release checkpoints under `./artifacts/` or pass explicit paths:

```bash
go run ./cmd/aurelius serve \
  -backend math-router \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -derivative-checkpoint ./artifacts/math-transformer-2layer-l7-derivative-full-v2.json
```

This route is intended for precise supported math tasks, not unrestricted general chat. It is appropriate for arithmetic, multiplication, and the derivative formats covered by the training and router code.

## Checkpoint Release

Trainer checkpoints include Adam optimizer state so training can resume. Public inference artifacts should strip that state before release:

```bash
go run ./cmd/aurelius export-checkpoint \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -output ./release-checkpoints/math-router-arithmetic-v4b.json

go run ./cmd/aurelius export-checkpoint \
  -checkpoint ./artifacts/math-transformer-2layer-l7-derivative-full-v2.json \
  -output ./release-checkpoints/math-router-derivative-full-v2.json
```

Attach selected files from `./release-checkpoints/` to a GitHub Release. Do not commit or publish the full `artifacts/` directory; it can contain intermediate checkpoints, eval errors, datasets, and local experiment outputs.

Recommended release checks:

```bash
go test ./...

rg -n "/Users|C:\\\\Users|private|secret|token|api[_-]?key|password|email|Bearer|BEGIN .* PRIVATE KEY" ./release-checkpoints
```

Expected string-scan matches are limited to model field names such as `token_embeddings`.

## Training Smoke Test

Generate a small dataset and train a transformer checkpoint:

```bash
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-smoke \
  -operations add,sub,mul \
  -levels 1,2,4 \
  -train-count 2000 \
  -val-count 300

go run ./cmd/aurelius train-math \
  -model transformer \
  -data-dir ./data/arithmetic-smoke \
  -checkpoint ./artifacts/math-transformer-smoke.json \
  -context-size 32 \
  -embedding-dim 64 \
  -hidden-dim 256 \
  -num-heads 4 \
  -num-layers 2 \
  -max-steps 1000 \
  -log-every 100 \
  -grad-clip 1
```

## Architecture Overview

Aurelius keeps the runtime small and inspectable:

- `internal/tensor` provides basic CPU tensor operations.
- `internal/tokenizer` defines tokenizer boundaries and includes byte-level and GPT-2 style BPE tokenizers.
- `internal/model` defines shared model interfaces and configuration.
- `internal/arithmetic` generates synthetic math datasets and converts examples into training sequences.
- `internal/textdata` loads, inspects, deduplicates, splits, and converts raw text and instruction JSONL.
- `internal/gpt2` loads GPT-2 model config metadata from local `config.json` files.
- `internal/mathlm` contains the trainable autoregressive MLP and transformer language models.
- `internal/mathrouter` normalizes supported math prompts and coordinates model-first inference with deterministic fallback.
- `internal/transformer` contains a deterministic toy transformer-style model that exercises the runtime path.
- `internal/sampler` provides greedy and temperature-based next-token selection.
- `internal/runtime` coordinates tokenization, model forward passes, autoregressive generation, stop strings, and optional KV-cached decoding.
- `internal/server` serves the local chat interface and JSON generation API.
- `cmd/aurelius` exposes the prototype via a CLI.

## Package Layout

```text
aurelius/
  cmd/aurelius/        CLI entrypoint
  internal/tensor/     Basic tensor math primitives
  internal/tokenizer/  Tokenizer interfaces and implementations
  internal/gpt2/       GPT-2 asset config loader
  internal/model/      Shared model interfaces and config
  internal/transformer/ Tiny transformer-style model
  internal/mathrouter/ Math prompt normalization and router fallback
  internal/sampler/    Token sampling strategies
  internal/runtime/    Generation engine
  internal/server/     Web server and browser UI
  benchmarks/          Benchmark harnesses and notes
  docs/                Architecture and design notes
```

## Current Scope

Aurelius is strongest as a controlled math inference platform. The math-router backend combines trained specialist checkpoints with exact deterministic validation for supported expressions. The standalone checkpoints are still small neural models and should be evaluated with `eval-math` before being presented as general-purpose math solvers.

Generated datasets, checkpoints, cache directories, release checkpoint exports, and local virtual environments are excluded from Git. Public checkpoint files should be distributed through GitHub Releases or another artifact channel.

## Example CLI Usage

```bash
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius generate -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius generate-gpt2 -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -max-tokens 1
go run ./cmd/aurelius emit-gpt2-observation -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -top-k 5
go run ./cmd/aurelius inspect-gpt2-next -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -prompt "hello world" -top-k 5
go run ./cmd/aurelius validate-gpt2 -model-config /path/to/config.json -weights /path/to/model.safetensors -vocab /path/to/vocab.json -merges /path/to/merges.txt -fixture /path/to/reference.json
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic -levels 1,2,3,4,5
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l2-small-sub -operations sub -levels 2 -answer-digits 1 -small-difference-only
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l2-question -operations add,sub -levels 2 -templates question
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l3-worked -operations add,sub -levels 3 -reasoning-style worked
go run ./cmd/aurelius mix-math-data -output-dir ./data/arithmetic-l2-replay -inputs ./data/arithmetic-l2-transformer:1,./data/arithmetic-l2-small-sub:2
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l3-transformer -operations add,sub -levels 3
go run ./cmd/aurelius gen-math-instructions -data-dir ./data/arithmetic-l3-transformer -output-dir ./data/instructions/math
go run ./cmd/aurelius fetch-text-data -url-file ./data/text/math-urls.txt -output-dir ./data/text/web-math
go run ./cmd/aurelius inspect-text-data -text ./data/text/web-math
go run ./cmd/aurelius dedupe-text-data -text ./data/text/web-math -output-dir ./data/text/web-math-deduped
go run ./cmd/aurelius split-text-data -text ./data/text/web-math-deduped -output-dir ./data/text/web-math-split -val-ratio 0.1
go run ./cmd/aurelius train-math -data-dir ./data/arithmetic -checkpoint ./artifacts/mathlm.json
go run ./cmd/aurelius train-text -text ./data/text/web-math-split/train -val-text ./data/text/web-math-split/val -checkpoint ./artifacts/aurelius-text.json
go run ./cmd/aurelius eval-math -checkpoint ./artifacts/mathlm.json -data ./data/arithmetic/val.jsonl
go run ./cmd/aurelius eval-math -checkpoint ./artifacts/mathlm.json -data ./data/arithmetic/val.jsonl -show-errors 10 -errors-out ./artifacts/math-errors.json
go run ./cmd/aurelius generate-math -checkpoint ./artifacts/mathlm.json -prompt "12 + 7 = "
go run ./cmd/aurelius generate-checkpoint -checkpoint ./artifacts/aurelius-text.json -prompt "User: hello\n\nAssistant:" -max-tokens 64
go run ./cmd/aurelius export-checkpoint -checkpoint ./artifacts/mathlm.json -output ./release-checkpoints/mathlm-inference.json
go run ./cmd/aurelius serve
go run ./cmd/aurelius serve -backend gpt2
go run ./cmd/aurelius serve -backend mathlm -checkpoint ./artifacts/aurelius-text.json
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

`serve` supports `-backend auto|toy|gpt2|mathlm|math-router`. In `auto` mode, Aurelius uses `math-router` when both specialist checkpoints are provided, uses `mathlm` when a single JSON checkpoint is provided, uses GPT-2 when complete assets exist under `artifacts/gpt2/`, and otherwise falls back to the toy model.

The web UI provides:

- a chat-style interface
- prompt entry
- max token control
- temperature control
- top-k control
- cache usage toggle
- browser-side history persisted in `localStorage`
- basic markdown rendering for assistant responses

When `serve` is using the GPT-2 backend, the web path now applies conservative request limits for responsiveness: a short assistant preamble, bounded temperature and top-k sampling, modest default reply length, capped generation length, trimmed conversation history, and cache-aware incremental decoding.

When `serve` is using the `mathlm` backend, it loads a local Aurelius JSON checkpoint and applies chat-oriented defaults, including stop strings for `User:` turns. For higher-accuracy math web inference, `-backend math-router` normalizes user phrasing into direct math prompts and routes to arithmetic and derivative specialist checkpoints. See [docs/llm-training.md](docs/llm-training.md) for text pretraining, instruction tuning, checkpoint generation, web inference, and math regression commands.

## Arithmetic Training

Aurelius now includes a student-scale from-scratch training path for arithmetic. It uses:

- synthetic arithmetic JSONL datasets with curriculum metadata
- the existing byte tokenizer for a minimal training-first path
- `train-math -model mlp` for the original fixed-context autoregressive MLP language model
- `train-math -model transformer` for a configurable-depth causal decoder transformer with manual full-path backpropagation
- JSON checkpoints for save/resume
- step-limited training, progress logging, periodic checkpoints, gradient clipping, and learning-rate warmup/decay for longer curriculum runs
- direct-answer, worked-solution, compact worked, or derivative coefficient-vector completions via `gen-math-data -reasoning-style direct|worked|compact|coefficients`
- exact-match evaluation on held-out arithmetic prompts, grouped by operation and curriculum level

This path is intentionally small and inspectable. It is not a large-scale LLM training stack and it does not replace the existing GPT-2 inference path.

Curriculum levels let you scale data difficulty without changing the model first:

```bash
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l1 -operations add,sub -levels 1
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l1-l3 -operations add,sub -levels 1,2,3
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-l3-worked -operations add,sub -levels 3 -reasoning-style worked
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-word -operations word -levels 6
go run ./cmd/aurelius gen-math-data -output-dir ./data/arithmetic-derivative -operations derivative -levels 7 -reasoning-style coefficients
go run ./cmd/aurelius train-math -model transformer -data-dir ./data/arithmetic-l1 -checkpoint ./artifacts/math-transformer-l1.json -num-heads 4 -num-layers 2 -epochs 25 -batch-size 32 -learning-rate 0.003 -warmup-steps 100 -decay-steps 2000 -min-learning-rate 0.0003
```

For larger curriculum runs, prefer bounded smoke runs before committing to a long training job:

```bash
go run ./cmd/aurelius train-math -model transformer -data-dir ./data/arithmetic-l1-l3 -checkpoint ./artifacts/math-transformer-curriculum.json -num-heads 4 -num-layers 2 -epochs 50 -batch-size 32 -learning-rate 0.003 -warmup-steps 200 -decay-steps 5000 -min-learning-rate 0.0003 -max-steps 2000 -log-every 100 -save-every 1000 -grad-clip 1
go run ./cmd/aurelius train-math -model transformer -data-dir ./data/arithmetic-l1-l3 -checkpoint ./artifacts/math-transformer-curriculum.json -resume ./artifacts/math-transformer-curriculum.json -epochs 50 -batch-size 32 -learning-rate 0.003 -max-steps 2000 -log-every 100 -save-every 1000 -grad-clip 1
```

## Development Commands

```bash
gofmt -w ./cmd ./internal
go test ./...
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10
go run ./cmd/aurelius -prompt "hello world" -max-tokens 10 -use-cache=true
go run ./cmd/aurelius serve
```

## Notes

This repository supports Aurelius JSON checkpoints for trained math models and local GPT-2 style assets for parity experiments. The current focus is architectural clarity, stable tests, and a small inference path that future work can replace piece by piece.

Cache-aware generation is optional per model. `runtime.Engine` detects models that implement the cache-capable extension and uses incremental decoding only for those models; all other models continue to use the uncached full-sequence path.

The local web UI uses the same runtime engine and generation options as the CLI. Chat history is stored in the browser rather than persisted server-side.

The `generate-gpt2` path runs a non-cached GPT-2 style forward pass from loaded safetensors weights, while `inspect-gpt2-next` and `validate-gpt2` support parity validation. The default CLI and web UI still use the toy transformer unless a specific backend or checkpoint is provided.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
