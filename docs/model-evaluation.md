# Model Evaluation

Use two separate measurements:

- Raw checkpoint accuracy: tests the model in the exact prompt format it was trained on.
- Web/router accuracy: tests the user-facing chat path after prompt normalization, model-first decoding, and deterministic fallback.

## Raw Checkpoint Evals

Evaluate the arithmetic checkpoint against the broad level 1-4 validation set:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -data ./data/arithmetic-l1-l4-direct-balanced/val.jsonl \
  -max-tokens 4 \
  -show-errors 50 \
  -errors-out ./artifacts/math-transformer-2layer-l1-l4-direct-v4b-regression-errors.json
```

Evaluate multiplication only:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -data ./data/arithmetic-l4-direct/val.jsonl \
  -max-tokens 4 \
  -show-errors 50 \
  -errors-out ./artifacts/math-transformer-2layer-l4-direct-v4b-regression-errors.json
```

Evaluate derivative checkpoints separately:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/math-transformer-2layer-l7-derivative-full-v2.json \
  -data ./data/arithmetic-l7-derivative/val.jsonl \
  -max-tokens 24 \
  -show-errors 50 \
  -errors-out ./artifacts/math-transformer-2layer-l7-derivative-full-v2-regression-errors.json
```

## Web/Router Regression

The web path should be checked with:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go test ./internal/mathrouter
```

This covers exact-answer edge cases after natural-language normalization. In the web backend, the router tries the model first and returns the model answer when it exactly matches the deterministic answer; otherwise it falls back to the deterministic answer. Add new cases there whenever the website gives a surprising answer.

## Training On Mistakes

Use `gen-math-error-replay` to turn model misses into corrected replay data. This keeps the failed prompts but replaces the completion with the router-corrected answer when the router supports the expression, otherwise with the eval expected answer.

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius gen-math-error-replay \
  -errors ./artifacts/math-transformer-2layer-l1-l4-direct-v4b-errors.json \
  -output-dir ./data/arithmetic-l1-l4-v4b-error-replay \
  -repeat 8 \
  -val-ratio 0.1 \
  -seed 1
```

Mix the replay data back into the broad dataset so the model does not overfit only the misses:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius mix-math-data \
  -output-dir ./data/arithmetic-l1-l4-direct-balanced-v4b-replay \
  -inputs ./data/arithmetic-l1-l4-direct-balanced:1,./data/arithmetic-l1-l4-v4b-error-replay:8 \
  -seed 2
```

Fine-tune from the previous checkpoint with a small learning rate:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius train-math \
  -resume ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -data-dir ./data/arithmetic-l1-l4-direct-balanced-v4b-replay \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4c.json \
  -learning-rate 0.00005 \
  -warmup-steps 100 \
  -decay-steps 10000 \
  -min-learning-rate 0.000005 \
  -max-steps 20000 \
  -log-every 100 \
  -save-every 1000 \
  -grad-clip 1
```

After fine-tuning, rerun the raw eval and compare the same groups, especially `template[question]`, `level[3]`, and `operation[mul]`.

## Release Checkpoint Export

Do not publish the entire `artifacts/` directory. It can contain intermediate checkpoints, error dumps, and local experiment files. Publish only the selected inference checkpoints.

Export release checkpoints without Adam optimizer state:

```sh
env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius export-checkpoint \
  -checkpoint ./artifacts/math-transformer-2layer-l1-l4-direct-v4b.json \
  -output ./release-checkpoints/math-router-arithmetic-v4b.json

env GOCACHE=/Users/augustahmed/Aurelius/.gocache go run ./cmd/aurelius export-checkpoint \
  -checkpoint ./artifacts/math-transformer-2layer-l7-derivative-full-v2.json \
  -output ./release-checkpoints/math-router-derivative-full-v2.json
```

Before attaching these files to a GitHub Release, scan them for private strings:

```sh
rg -n "august|ahmed|/Users|Aurelius|http|https|private|secret|token|api[_-]?key|password|email|@" ./release-checkpoints
```

Expected result: no matches except model field names such as `token_embeddings`. The exported checkpoints should contain numeric weights, model config, and `adam: null`, not training examples or local paths.

## Edge Cases To Track

Arithmetic:

- `0 + 0`
- `99 + 99`
- `-4 plus 9`
- `10 - 99`
- `12 x 12`
- `12 × 6`
- `6 multiplied by 7`
- `1000 * 0`

Derivatives:

- `x^2`
- `3x + 2`
- `5`
- `-x^3 + 4x - 9`
- `x^3 - x`
- `2x^2 + 3x^2 + x`

Prompt wrappers:

- Bare prompt: `What is 7 * 8?`
- Chat prompt: `User: What is 7 * 8?\n\nAssistant:`
- Solve wording: `Can you solve 7 times 8?`
- Misspelled derivative wording: `What is the derrivative of x^2?`
