# Arithmetic Training

## Purpose

This workflow turns Aurelius into a small, student-scale language model project that can be trained from scratch on synthetic arithmetic data.

It is intentionally narrow:

- synthetic arithmetic data only
- byte-level tokenization
- fixed-context autoregressive MLP language model
- exact-match evaluation on held-out arithmetic prompts

This is not a frontier LLM training stack and it does not replace the existing GPT-2 inference path.

## Dataset Format

Datasets are written as JSONL with one example per line:

```json
{"prompt":"2 + 3 = ","completion":"5","operation":"add"}
{"prompt":"What is 12 * 4? ","completion":"48","operation":"mul"}
```

The generator writes:

```text
<output-dir>/
  train.jsonl
  val.jsonl
  meta.json
```

`meta.json` records operand ranges, split sizes, enabled operations, seed, and tokenizer choice.

## Training Workflow

Generate a dataset:

```bash
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic \
  -train-count 4000 \
  -val-count 500 \
  -min-operand 0 \
  -max-operand 20 \
  -operations add,sub,mul,div \
  -seed 1
```

Train a small model from scratch:

```bash
go run ./cmd/aurelius train-math \
  -data-dir ./data/arithmetic \
  -checkpoint ./artifacts/mathlm.json \
  -context-size 32 \
  -embedding-dim 32 \
  -hidden-dim 128 \
  -epochs 10 \
  -batch-size 64 \
  -learning-rate 0.01 \
  -seed 1
```

Resume training:

```bash
go run ./cmd/aurelius train-math \
  -data-dir ./data/arithmetic \
  -checkpoint ./artifacts/mathlm.json \
  -resume ./artifacts/mathlm.json \
  -epochs 5
```

## Evaluation

Evaluate on held-out examples:

```bash
go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/mathlm.json \
  -data ./data/arithmetic/val.jsonl
```

The evaluator reports exact-match accuracy on generated completions.

## Inference

Run a prompt against the trained checkpoint:

```bash
go run ./cmd/aurelius generate-math \
  -checkpoint ./artifacts/mathlm.json \
  -prompt "12 + 7 = " \
  -max-tokens 8
```

## Model Shape

The arithmetic trainer currently uses:

- byte tokenizer with vocabulary size `256`
- fixed left-padded context window
- embedding lookup
- one hidden MLP layer with `tanh`
- output projection to next-token logits
- Adam optimizer

This is a real autoregressive model and it really trains from random initialization, but it is not yet a transformer training system.

## Limitations

- byte-level tokenization is simple, not efficient
- the model is a small MLP LM, not a decoder-only transformer with training support
- arithmetic quality depends heavily on operand range and dataset size
- exact-match accuracy is useful for arithmetic but not a general language-model benchmark
- the current training loop is CPU-first and educational rather than optimized

## What “Useful for Basic Math” Means Here

At this stage, “useful” means:

- loss decreases during training
- held-out arithmetic accuracy improves after training
- the trained checkpoint can complete short arithmetic prompts better than random initialization

It does not mean broad reasoning ability or frontier chat quality.
