# LLM Training Path

Aurelius can now train byte-tokenized transformer checkpoints on raw text and instruction-style JSONL. This is a small local LLM path, not a production-scale model recipe.

## Raw Text Pretraining

Fetch web pages into local cleaned text first:

```sh
go run ./cmd/aurelius fetch-text-data \
  -urls "https://example.com/math-lesson,https://example.com/calculus-notes" \
  -output-dir ./data/text/web-math \
  -max-pages 20 \
  -max-bytes 2097152 \
  -timeout 15s
```

For larger source lists:

```sh
go run ./cmd/aurelius fetch-text-data \
  -url-file ./data/text/math-urls.txt \
  -output-dir ./data/text/web-math
```

Only fetch pages you are allowed to use. The command writes cleaned `.txt` files plus `sources.jsonl`; training should read the local output directory, not live URLs.

Inspect the fetched text before training:

```sh
go run ./cmd/aurelius inspect-text-data \
  -text ./data/text/web-math
```

Remove duplicate paragraphs and very short fragments:

```sh
go run ./cmd/aurelius dedupe-text-data \
  -text ./data/text/web-math \
  -output-dir ./data/text/web-math-deduped \
  -min-paragraph-runes 20
```

Create a deterministic train/validation split:

```sh
go run ./cmd/aurelius split-text-data \
  -text ./data/text/web-math-deduped \
  -output-dir ./data/text/web-math-split \
  -val-ratio 0.1 \
  -seed 1
```

Use `train-text` with one or more `.txt` or `.md` files or directories:

```sh
go run ./cmd/aurelius train-text \
  -text ./data/text/web-math-split/train \
  -val-text ./data/text/web-math-split/val \
  -checkpoint ./artifacts/aurelius-text.json \
  -context-size 128 \
  -embedding-dim 128 \
  -hidden-dim 512 \
  -num-heads 4 \
  -num-layers 4 \
  -batch-size 16 \
  -learning-rate 0.0003 \
  -warmup-steps 100 \
  -decay-steps 10000 \
  -min-learning-rate 0.00003 \
  -max-steps 20000 \
  -log-every 100 \
  -save-every 1000 \
  -grad-clip 1
```

Directory inputs are expanded to `.txt` and `.md` files. Text training builds next-token prediction sequences with `-stride`; larger strides are faster but provide fewer examples.

## Instruction Tuning

Instruction files are JSONL. Each line can use either explicit prompt/completion:

```json
{"prompt":"User: What is 2 + 2?\n\nAssistant:","completion":"4"}
```

or instruction fields:

```json
{"system":"Be concise.","instruction":"What is 2 + 2?","output":"4"}
```

You can convert generated arithmetic datasets into instruction examples:

```sh
go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-instruct-source \
  -operations add,sub,mul,derivative \
  -levels 1,2,3,4,7 \
  -templates all \
  -reasoning-style direct \
  -train-count 20000 \
  -val-count 2000

go run ./cmd/aurelius gen-math-instructions \
  -data-dir ./data/arithmetic-instruct-source \
  -output-dir ./data/instructions/math \
  -format compact
```

Resume from a pretrained checkpoint:

```sh
go run ./cmd/aurelius train-text \
  -instructions ./data/instructions/math/train.jsonl \
  -val-instructions ./data/instructions/math/val.jsonl \
  -resume ./artifacts/aurelius-text.json \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -batch-size 16 \
  -learning-rate 0.0001 \
  -warmup-steps 50 \
  -decay-steps 5000 \
  -min-learning-rate 0.00001 \
  -max-steps 10000 \
  -log-every 100 \
  -save-every 1000 \
  -grad-clip 1
```

## Generation

Use a local checkpoint directly:

```sh
go run ./cmd/aurelius generate-checkpoint \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -prompt $'User: What is 7 * 8?\n\nAssistant:' \
  -max-tokens 64 \
  -temperature 0.7 \
  -top-k 40 \
  -stop '\nUser:,\n\nUser:'
```

Use greedy decoding with `-temperature 0 -top-k 0`. Use `-top-k 40 -temperature 0.7` for more varied chat output.

## Instruction Eval

Evaluate instruction-tuned checkpoints against the same prompt wrapper used for training:

```sh
go run ./cmd/aurelius eval-instructions \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -instructions ./data/instructions/math/val.jsonl \
  -show-errors 20 \
  -errors-out ./artifacts/aurelius-instruct-errors.json
```

## Web Inference

Serve a checkpoint through `/generate`:

```sh
go run ./cmd/aurelius serve \
  -backend mathlm \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -addr localhost:8080
```

The JSON API accepts `prompt`, `messages`, `max_tokens`, `temperature`, `top_k`, `use_cache`, and `stop`.

For the current math-centered path, prefer the deterministic router instead of instruction-tuning one checkpoint to understand every chat wrapper. It normalizes user text into the direct prompt format, solves supported integer arithmetic and polynomial derivatives directly, and routes unsupported forms to specialist checkpoints:

```sh
go run ./cmd/aurelius serve \
  -backend math-router \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -derivative-checkpoint ./artifacts/math-transformer-2layer-l7-derivative-full-v2.json \
  -addr localhost:8080
```

Examples accepted by the router include `What is 7 * 8?`, `Can you solve 7 times 8?`, and `What is the derivative of x^2?`.

## Math Regression Evals

Keep evaluating math datasets after text or instruction tuning:

```sh
go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -data ./data/arithmetic-l1-l4-direct/val.jsonl \
  -max-tokens 4 \
  -show-errors 20 \
  -errors-out ./artifacts/aurelius-instruct-math-errors.json
```
