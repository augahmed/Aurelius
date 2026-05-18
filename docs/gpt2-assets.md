# GPT-2 Asset Loading Slice

## Decision

The next vertical slice toward real inference loads real GPT-2 style assets from disk before attempting weight-backed forward passes.

This slice includes:

- GPT-2 BPE tokenizer loading from `vocab.json` and `merges.txt`
- GPT-2 model config loading from `config.json`
- CLI commands that let contributors validate local assets without wiring them into the current toy transformer

## Why This Slice

The current blocker is not the chat UI. It is the lack of real text preprocessing and the lack of any path for external model metadata to enter the system.

Adding tokenizer and config loading first creates a stable boundary for later work:

- token ids can now match a real GPT-2 vocabulary instead of raw bytes
- model dimensions can come from actual config files instead of hard-coded toy defaults
- future weight loading can validate tensor shapes against parsed config data

## What This Does Not Do Yet

- load GPT-2 weight tensors
- execute a GPT-2 forward pass
- swap the chat server away from the toy transformer

That separation is intentional. It keeps this change small, testable, and honest about what the runtime can and cannot do today.
