# ADR-0004: Initial compression operators: redundancy pruning and L2-norm KV eviction

- **Status:** Accepted (operators are placeholders; stronger baselines are on the roadmap)
- **Date:** 2026-11-10

## Context
The oracle study needs compression operators that are cheap, deterministic, and compatible with
SDPA/flash attention, so no attention matrices are materialized. Our question is about *allocation*
(how much to compress each request), not about inventing a new pruning rule.

## Decision
**Visual tokens** (applied before prefill; text tokens never dropped):
- `uniform`: evenly spaced tokens in raster order.
- `dissim`: drop the tokens most cosine-similar to the image's mean embedding (background or redundant
  patches). Query-agnostic, one matmul.

**KV cache** (applied after prefill, same budget in every layer):
- L2-norm eviction (Devoto et al., 2024): keep keys with the **lowest** L2 norm, which correlate with high
  attention. Attention-free, and RoPE-invariant (RoPE is a rotation).
- Always keep the first 4 tokens (attention sinks) and the last 16 (question tail plus generation prompt).

Grid: visual keep {1.0, 0.75, 0.5, 0.25} × {uniform, dissim} × KV keep {1.0, 0.75, 0.5, 0.25}
= 28 configs (the 1.0 visual setting is shared by both methods).

## Alternatives considered
- **FastV / SparseVLM / attention-score pruning.** Stronger, but need attention weights (eager attention)
  or mid-layer surgery. These are the planned v0.2 baselines.
- **SnapKV / H2O / PyramidKV.** Need attention scores. Planned v0.2 baselines.

## Consequences
- The absolute quality of each config is lower than with state-of-the-art operators. The *relative*
  question (per-request vs static allocation) is still valid, but must be re-checked with stronger
  operators before any paper claim.
- Operators are plain functions (`uniinfer.compression`), so new ones are easy first contributions.
