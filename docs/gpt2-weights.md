# GPT-2 Weight Format

## Decision

The first real GPT-2 inference path uses local `safetensors` weight files plus the already-supported `config.json`, `vocab.json`, and `merges.txt` assets.

## Why `safetensors`

- It is a real model weight format already used in the GPT-2 ecosystem.
- It is simple enough to parse directly with the Go standard library.
- It avoids introducing a heavyweight dependency just to load tensors.
- It gives Aurelius a path toward loading real exported GPT-2 checkpoints instead of a toy JSON-only format.

## Current Scope

The loader currently supports:

- local `.safetensors` files
- `F32` and `F64` tensor payloads
- GPT-2 tensor names such as:
  - `transformer.wte.weight`
  - `transformer.wpe.weight`
  - `transformer.h.N.attn.c_attn.weight`
  - `transformer.h.N.attn.c_proj.weight`
  - `transformer.h.N.mlp.c_fc.weight`
  - `transformer.h.N.mlp.c_proj.weight`
  - `transformer.ln_f.weight`

## Limitations

- No KV cache yet
- No quantized weights
- No partial tensor loading or mmap
- No support yet for the full range of safetensors dtypes

Those are follow-up steps. The goal of this slice is correctness-first GPT-2 forward inference with loaded weights.
