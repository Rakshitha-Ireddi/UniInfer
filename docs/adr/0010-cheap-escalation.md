# ADR-0010: Cheap escalation for KV cascades (one prefill, parked escalation cache)

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
Two results point at the same bottleneck. In v0.2 (ADR-0008) and again on the SnapKV grid, a confidence
cascade beats its shuffled-confidence control only when a retry does not pay for the failed attempt (the
"max" cost model). And signals available *before* answering (request features up to a prefill hidden state)
do not beat a task label, while the cheap answer's own confidence detects damage (AUROC 0.83). So
per-request adaptation should be a cascade, and the retry must be cheap in both memory and latency.
Until now, the "max" cost model was an accounting assumption. A real retry re-ran the vision encoder and
the prefill.

## Decision
1. **Restrict escalation to the KV axis.** Visual-token pruning happens before prefill, so changing it on a
   retry needs a new prefill. KV compression happens after prefill, so both cascade stages can come from
   one prefill. Cascades used for serving keep all visual tokens and vary only the SnapKV ratio.
2. **Park the escalation cache in pinned CPU memory.** After the single prefill, build the stage-2 cache
   (larger ratio) and copy it to pinned host memory. Then build the stage-1 cache on the GPU and free the
   full prefill KV. On escalation, free the stage-1 cache and copy stage 2 back over PCIe. GPU-resident
   decode KV never exceeds the stage being decoded, which makes the "max" cost model real.
3. **Correctness gate (`experiments/escalation/run.py --check`).** With escalation forced on and off, both
   the offload path and the naive re-prefill path must return exactly the single-config answers for
   stage 1 and stage 2.
4. **Evaluate on new requests.** Cascades and the static baseline are fit on the 450 v0.2 requests
   (`plan.py`) and run end to end on requests drawn with a different shuffle seed, skipping every request
   used before. We report score, realized KV cost (sum and max), escalation rate, GPU-resident KV bytes and
   wall-clock time per stage, for offload vs re-prefill.

## Alternatives considered
- **Keep the full prefill KV on the GPU and re-compress on demand.** Simplest, but the full KV stays
  resident for the whole decode, which defeats the memory saving.
- **Nested SnapKV selection (stage 2 ⊇ stage 1) and transfer only the difference.** Less PCIe traffic, but
  `topk` tie-breaking under max-pooling does not guarantee nesting. The stage-2 answer would then differ
  from the single-config answer, breaking the gate. Possible later, with a stable ranking.
- **Recompute from offloaded *full* KV.** Most flexible (any r2), but it moves the whole cache over PCIe.

## Consequences
- Each request costs one extra stage-2 compression and a host copy, even when it does not escalate. That
  cost is measured, not assumed.
- Host memory holds one stage-2 cache per in-flight request. At batch 1 this is negligible; under
  concurrent serving it becomes a pool-sizing question for v0.3.
- Visual-token pruning cannot be part of a cheap cascade; the static visual config is chosen offline.
