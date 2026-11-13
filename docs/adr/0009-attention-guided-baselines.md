# ADR-0009: Attention-guided baselines (SnapKV, FastV-style) on SDPA, via attention reconstruction

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
The v0.1 and v0.2 headroom results use simple operators: uniform or redundancy visual pruning, and
L2-norm KV eviction (ADR-0004). The obvious objection is that a *stronger* static operator might close the
gap between the best static config and the per-request oracle. The standard strong operators are guided by
attention: SnapKV (KV) and FastV (visual tokens). Both need attention weights. Our runtime uses SDPA, which
does not return them, and switching to eager attention would change numerics and break the equivalence gate
(ADR-0003).

## Decision
1. **Reconstruct the attention we need instead of materializing it.** Read-only forward pre-hooks store the
   RoPE'd queries of the last W = 16 prompt positions (the observation window) in every layer during
   prefill. The rotated keys are already in the KV cache, so `window_attention(q, K)` rebuilds the window's
   causal attention over all L tokens in O(W·L) per layer. Model outputs are unchanged.
2. **Correctness gate (`sweep.py --check-attn`).** Rebuild each layer's attention *output*
   (`o_proj(P · V)`) from the reconstructed weights and the cached values. Compare it with the output the
   model actually produced for the window. It must match within 2% relative error at every layer.
   Observed: 0.2–0.35% on TextVQA/DocVQA and up to 1.2% on MMBench (shorter prompts, BF16 rounding).
3. **SnapKV:** for each layer and KV head, sum attention over the window, average over the query heads of
   each GQA group, max-pool (kernel 7), and keep the top tokens plus the window. Every head keeps the same
   number of tokens (the KV-cost model is unchanged), but different heads may keep different tokens.
4. **FastV-style visual pruning:** rank image tokens by the attention they receive at layer 2 from the
   observation window (mean over heads), then prune them *before* prefill. FastV itself prunes inside the
   forward pass after layer K = 2, keeping full KV for layers 0–2. We prune every layer so that cost stays
   "relative KV memory" (ADR-0005) and both visual operators are comparable. The score needs a prefill of
   layers 0–2 over the full sequence (3/36 of a prefill for the 3B model). The sweep reuses the full
   prefill it already runs.

## Alternatives considered
- **Eager attention with `output_attentions=True`.** Exact, but changes numerics relative to SDPA, and the
  full L×L weights for 36 layers cost memory. Rejected; the reconstruction gives the same rows at a tiny
  cost.
- **Exact FastV (prune mid-forward).** More faithful, but needs a modified decoder loop and mixes two KV
  lengths in one cache. It may be added later as a separate operator. The deviation is documented here and
  in the README.
- **Observation window = question tokens only.** The last 16 positions already contain the question tail
  and the generation prompt (the same window that L2-norm eviction always keeps). Simpler and consistent.

## Consequences
- The baseline sweep grows from 28 to 70 configs per request (visual {uniform, dissim, attn} × KV
  {L2-norm, SnapKV}).
- Every result on attention-guided operators depends on the `--check-attn` gate passing on the GPU used.
- In deployment, FastV-style pruning adds a shallow pass. Its latency cost is not counted under the
  KV-memory cost model, which is the same simplification as for the selector's features (ADR-0005).

## Outcome (2026-11-13)
Both gates passed on an RTX A5000. The 70-config sweep plus a 30-config low-ratio SnapKV sweep reproduce
every prediction they share with v0.1. SnapKV is close to lossless down to 25% KV and removes most of the
oracle-static gap at budgets of 15% and above; the gap survives at 10% and below. Per-request allocation
is therefore evaluated against SnapKV, at tight budgets, from now on
([results](../../experiments/baselines/README.md)).
