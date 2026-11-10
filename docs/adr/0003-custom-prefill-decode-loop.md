# ADR-0003: Own the prefill/decode loop instead of `generate()`

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
Qwen2.5-VL uses **M-RoPE**: every token has a 3-D (temporal, height, width) position id, computed by
`get_rope_index` from the full token sequence. If we drop visual tokens from `input_ids` and call
`generate()`, the model recomputes positions for the shorter sequence. Every surviving patch then gets
the wrong spatial position. Swapping in a compressed KV cache mid-generation is also not exposed by
`generate()`.

## Decision
`QwenVLRunner` (`src/uniinfer/runtime/qwen25vl.py`) runs the stages explicitly:
1. **encode**: vision encoder once, then full-length embeddings and full-length M-RoPE ids.
2. **prefill**: on the *kept* tokens only, passing their *original* position ids explicitly
   (`position_ids` given, so HF skips its own RoPE computation).
3. **compress**: build a new KV cache from the prefill cache (the original is not mutated, so many
   KV ratios share one prefill).
4. **decode**: greedy. Step *t* uses position `max(prompt positions) + 1 + t`, exactly what HF does via
   `rope_deltas`.

**Correctness gate:** `sweep.py --check` asserts that the uncompressed path is token-identical to
`model.generate()` on samples from every dataset. This passes on all checked samples.

We pin `transformers>=4.49,<4.52` because the code uses `DynamicCache.key_cache` / `value_cache`,
which changed in later releases.

## Alternatives considered
- **Logits processors or hooks inside `generate()`.** Can't fix positions after pruning, and fragile.
- **Attention masking instead of removing tokens.** Correct positions, but saves no compute or memory,
  which defeats the purpose.

## Consequences
- Full control, and the experiment reuses work: 1 encoder pass, 7 prefills and 28 decodes per sample.
- Batch size is 1 for now. Batched decode with per-request budgets is roadmap v0.3.
- Upgrading transformers requires porting the cache access. CI pins the version.
