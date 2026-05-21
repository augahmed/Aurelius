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
{"prompt":"2 + 3 = ","completion":"5","operation":"add","level":1,"min_operand":0,"max_operand":9,"answer_digits":1,"template":"equation"}
{"prompt":"What is 12 * 4? ","completion":"48","operation":"mul","level":4,"min_operand":0,"max_operand":12,"answer_digits":2,"template":"question"}
```

The generator writes:

```text
<output-dir>/
  train.jsonl
  val.jsonl
  meta.json
```

`meta.json` records operand ranges, split sizes, enabled operations, enabled curriculum levels, answer filters, seed, and tokenizer choice.

## Curriculum Levels

The generator supports explicit levels:

- `1`: single-digit addition and subtraction
- `2`: two-digit addition and subtraction without carry or borrow
- `3`: two-digit addition and subtraction with carry or borrow
- `4`: small multiplication tables
- `5`: exact integer division
- `6`: simple two-step word problems

Each generated example includes metadata such as `level`, `operation`, `answer_digits`, `small_difference`, `requires_carry`, `requires_borrow`, and `template`. Train and validation data cycle through compatible level/operation pairs before shuffling, so held-out examples preserve the same coverage for grouped metrics.

Target known weak spots with answer-shape filters:

```bash
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-l2-small-sub \
  -train-count 10000 \
  -val-count 1000 \
  -operations sub \
  -levels 2 \
  -answer-digits 1 \
  -small-difference-only \
  -seed 7
```

Mix normal and targeted datasets for replay training:

```bash
go run ./cmd/aurelius mix-math-data \
  -output-dir ./data/arithmetic-l2-replay \
  -inputs ./data/arithmetic-l2-transformer:1,./data/arithmetic-l2-small-sub:2 \
  -seed 1
```

`mix-math-data` expects each input to be a generated dataset directory containing `train.jsonl` and `val.jsonl`. Integer weights repeat a source before deterministic shuffling, so `targeted:2` contributes twice as many examples as `targeted:1`. The output is a normal dataset directory that works with `train-math` and `eval-math`.

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
  -levels 1,2,3,4,5 \
  -seed 1
```

Generate only the easiest curriculum levels:

```bash
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-l1-l2 \
  -train-count 3000 \
  -val-count 500 \
  -operations add,sub \
  -levels 1,2 \
  -seed 1
```

Generate word problems:

```bash
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-word \
  -train-count 2000 \
  -val-count 300 \
  -operations word \
  -levels 6 \
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

Train the tiny transformer backend:

```bash
go run ./cmd/aurelius train-math \
  -model transformer \
  -data-dir ./data/arithmetic \
  -checkpoint ./artifacts/math-transformer.json \
  -context-size 32 \
  -embedding-dim 32 \
  -hidden-dim 128 \
  -num-heads 4 \
  -epochs 25 \
  -batch-size 32 \
  -learning-rate 0.003
```

## Evaluation

Evaluate on held-out examples:

```bash
go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/mathlm.json \
  -data ./data/arithmetic/val.jsonl
```

The evaluator reports exact-match accuracy on generated completions.

It also reports grouped accuracy:

```text
accuracy=0.1000 correct=5 total=50 max_tokens=3
operation[add]=0.1250 correct=3 total=24
operation[sub]=0.0769 correct=2 total=26
level[1]=0.1000 correct=5 total=50
```

For failure analysis, collect incorrect examples:

```bash
go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/math-transformer-l2.json \
  -data ./data/arithmetic-l2/val.jsonl \
  -show-errors 20 \
  -errors-out ./artifacts/math-transformer-l2-errors.json
```

`-show-errors` prints the first `N` incorrect examples with prompt, expected completion, generated completion, operation, level, template, carry/borrow flags, and operand-range metadata. `-errors-out` writes all incorrect examples plus grouped template counts as JSON. Without these flags, `eval-math` keeps the normal concise output.
Debug output also groups by `answer_digits` and `small_difference`, which is useful for detecting digit-length failures such as two-digit subtraction producing one-digit answers.

Use these grouped metrics to decide when to add harder levels. A practical starting sequence is:

1. train on levels `1`
2. train on levels `1,2`
3. train on levels `1,2,3`
4. add multiplication with level `4`
5. add division with level `5`
6. add word problems with level `6`

Training uses each prompt as context and optimizes only the answer plus trailing newline. This keeps the small model focused on answer prediction instead of spending most updates learning to reproduce prompt text.

## Inference

Run a prompt against the trained checkpoint:

```bash
go run ./cmd/aurelius generate-math \
  -checkpoint ./artifacts/mathlm.json \
  -prompt "12 + 7 = " \
  -max-tokens 8
```

## Model Shape

The arithmetic trainer supports two backends:

`-model mlp` uses:

- byte tokenizer with vocabulary size `256`
- fixed left-padded context window
- embedding lookup
- one hidden MLP layer with `tanh`
- output projection to next-token logits
- Adam optimizer

`-model transformer` uses:

- byte tokenizer with vocabulary size `256`
- token and positional embeddings
- one causal self-attention block
- layer norms
- residual connections
- MLP block with GELU
- output projection to next-token logits

The transformer backend now backpropagates through the full one-block path: embeddings, positional embeddings, layer norms, Q/K/V attention weights, attention output projection, MLP weights, and output head. It is still a small educational CPU implementation rather than an optimized training stack.

## Limitations

- byte-level tokenization is simple, not efficient
- the transformer backend is only one decoder block and trains slowly on CPU
- the transformer implementation favors readable manual gradients over speed or memory efficiency
- arithmetic quality depends heavily on operand range and dataset size
- exact-match accuracy is useful for arithmetic but not a general language-model benchmark
- the current training loop is CPU-first and educational rather than optimized

## What “Useful for Basic Math” Means Here

At this stage, “useful” means:

- loss decreases during training
- held-out arithmetic accuracy improves after training
- the trained checkpoint can complete short arithmetic prompts better than random initialization

It does not mean broad reasoning ability or frontier chat quality.
