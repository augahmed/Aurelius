# GPT-2 Parity Validation

## Goal

The current GPT-2 path must be validated against a reference fixture before it can replace the toy runtime path or serve as a base for KV cache work.

## Current Harness

Aurelius now includes:

- `validate-gpt2` to load:
  - `config.json`
  - `vocab.json`
  - `merges.txt`
  - `model.safetensors`
  - a parity fixture JSON file
- `inspect-gpt2-next` to print the top next-token candidates for a prompt
- `emit-gpt2-observation` to export Aurelius's own observed prompt tokens and top next-token predictions in fixture-compatible JSON
- a checked-in example fixture at `internal/gpt2/testdata/tiny_parity_fixture.json`

## Expected File Layout

```text
my-gpt2-checkpoint/
  config.json
  vocab.json
  merges.txt
  model.safetensors
  fixtures/
    prompt-1.json
    prompt-2.json
```

## Fixture Schema

```json
{
  "name": "fixture-name",
  "prompt": "prompt text",
  "expected_input_tokens": [1, 2, 3],
  "expected_top_tokens": [
    {"token": 42, "logit": 12.345}
  ],
  "expected_logits": [],
  "logit_tolerance": 0.00001,
  "metadata": {}
}
```

Use `expected_logits` when you want full next-token parity. Use `expected_top_tokens` when you only need stable top-token checks.

## External Reference Workflow

1. Export a real GPT-2 checkpoint into one directory containing:
   - `config.json`
   - `vocab.json`
   - `merges.txt`
   - `model.safetensors`
2. Choose one or more fixed prompts.
3. In a known-good external implementation, record:
   - encoded prompt tokens
   - either full next-token logits or stable top-k next tokens and logits
4. Save those values into a fixture JSON file matching the schema above.
5. Run Aurelius validation:

```bash
go run ./cmd/aurelius validate-gpt2 \
  -model-config /path/to/config.json \
  -weights /path/to/model.safetensors \
  -vocab /path/to/vocab.json \
  -merges /path/to/merges.txt \
  -fixture /path/to/reference.json
```

6. If validation fails, inspect Aurelius output locally with:

```bash
go run ./cmd/aurelius inspect-gpt2-next \
  -model-config /path/to/config.json \
  -weights /path/to/model.safetensors \
  -vocab /path/to/vocab.json \
  -merges /path/to/merges.txt \
  -prompt "your prompt" \
  -top-k 10
```

7. If you want a local Aurelius-side observation snapshot for debugging, emit one with:

```bash
go run ./cmd/aurelius emit-gpt2-observation \
  -model-config /path/to/config.json \
  -weights /path/to/model.safetensors \
  -vocab /path/to/vocab.json \
  -merges /path/to/merges.txt \
  -prompt "your prompt" \
  -top-k 10
```

## Next External Step

The checked-in tiny fixture proves the Aurelius-side parity harness works, but it is not a real exported GPT-2 checkpoint comparison.

The next external validation step is:

1. Export a real GPT-2 checkpoint with:
   - `config.json`
   - `vocab.json`
   - `merges.txt`
   - `model.safetensors`
2. Produce a reference fixture from a known-good implementation for one or more fixed prompts.
3. Run `validate-gpt2` against those assets and fixture files.

Use `emit-gpt2-observation` only for local debugging. Do not treat Aurelius-generated observations as the external reference.

Only after that parity is established should Aurelius promote GPT-2 closer to the default engine path or add KV cache support.
