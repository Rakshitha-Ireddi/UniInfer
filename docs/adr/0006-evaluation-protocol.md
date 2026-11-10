# ADR-0006: Evaluation protocol: oracle first, request-grouped CV, luck-free claims

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
Selecting the best of 28 configs per request with true scores inflates results. A degraded config
sometimes answers correctly *by chance* when the uncompressed model was wrong (7.8% of our requests).
Learned selectors can also leak information if tuned on test requests.

## Decision
1. **Oracle first.** Measure the per-request upper bound before building any controller. If the gap to
   static is small, stop (ADR-0001).
2. **Luck-free oracle for claims.** Cap each request's score at its uncompressed score
   (`S_capped = min(S, S[:, full])`). Raw-oracle numbers are reported only alongside it.
3. **Request-grouped K-fold CV** (5 folds × 3 seeds). Static configs, per-task configs, λ and the
   selector are all fit on training folds only. Everything is scored on held-out requests with **real**
   scores, so luck affects every method equally.
4. **Baselines that must be beaten:**
   - *static*: one config for all requests;
   - *per-task*: one config per dataset, each dataset meeting the budget on its own;
   - *task-aware allocation*: selector trained on the dataset label only. This shares budget across task
     types, and it is the baseline that per-request features must beat.
5. **Latency is diagnostic only** until measured on a dedicated GPU (our GPUs are shared).

## Consequences
- Our headline numbers are conservative.
- The task-aware baseline turned out to be strong (see results). This prevents us from attributing gains
  to per-request features when they come from per-task reallocation.
