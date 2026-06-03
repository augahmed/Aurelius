# LLM Training Path

Aurelius can now train byte-tokenized transformer checkpoints on raw text and instruction-style JSONL. This is a small local LLM path, not a production-scale model recipe.

## Raw Text Pretraining

Fetch web pages into local cleaned text first:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius fetch-text-data \
  -urls "https://example.com/math-lesson,https://example.com/calculus-notes" \
  -output-dir ./data/text/web-math \
  -max-pages 20 \
  -max-bytes 2097152 \
  -timeout 15s
```

For larger source lists:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius fetch-text-data \
  -url-file ./data/text/math-urls.txt \
  -output-dir ./data/text/web-math
```

Only fetch pages you are allowed to use. The command writes cleaned `.txt` files plus `sources.jsonl`; training should read the local output directory, not live URLs.

Use `train-text` with one or more `.txt` or `.md` files or directories:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius train-text \
  -text ./data/text/web-math \
  -val-text ./data/text/val \
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

Resume from a pretrained checkpoint:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius train-text \
  -instructions ./data/instructions/train.jsonl \
  -val-instructions ./data/instructions/val.jsonl \
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
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius generate-checkpoint \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -prompt $'User: What is 7 * 8?\n\nAssistant:' \
  -max-tokens 64 \
  -temperature 0.7 \
  -top-k 40 \
  -stop '\nUser:,\n\nUser:'
```

Use greedy decoding with `-temperature 0 -top-k 0`. Use `-top-k 40 -temperature 0.7` for more varied chat output.

## Web Inference

Serve a checkpoint through `/generate`:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius serve \
  -backend mathlm \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -addr localhost:8080
```

The JSON API accepts `prompt`, `messages`, `max_tokens`, `temperature`, `top_k`, `use_cache`, and `stop`.

## Math Regression Evals

Keep evaluating math datasets after text or instruction tuning:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/aurelius-instruct.json \
  -data ./data/arithmetic-l1-l4-direct/val.jsonl \
  -max-tokens 4 \
  -show-errors 20 \
  -errors-out ./artifacts/aurelius-instruct-math-errors.json
```
