# Aurelius Architecture

## High-Level Diagram

```text
prompt text
    |
    v
tokenizer.Encode
    |
    v
token ids
    |
    v
runtime.Engine
    |
    +--> model.Forward(tokens, cache)
    |         |
    |         v
    |   transformer.TinyTransformer
    |         |
    |         +--> embeddings
    |         +--> transformer blocks
    |         +--> output projection
    |         v
    |       logits
    |
    +--> sampler.Sample(logits)
    |
    v
generated token ids
    |
    v
tokenizer.Decode
    |
    v
generated text
```

## Current Prototype Design

The prototype is CPU-only and correctness-first. It uses a deterministic toy transformer-style model so the full inference loop can be exercised without introducing weight files, external model formats, or unstable randomness.

Current implementation boundaries:

- `internal/model` owns model configuration and forward-pass contracts.
- `internal/transformer` implements a tiny model with deterministic embeddings, residual blocks, and an output projection.
- `internal/runtime` owns the autoregressive loop.
- `internal/tokenizer` currently uses a byte-level tokenizer for simplicity and full input coverage.
- `internal/tensor` exposes the minimal operations needed for the tiny model and future incremental expansion.
- `internal/server` provides API skeletons without committing to a transport design too early.

## Future Milestones

- Real BPE tokenizer
- GPT-2 weight loading
- Attention implementation
- KV cache
- Streaming API
- Benchmarking
- Batching
- Quantization
