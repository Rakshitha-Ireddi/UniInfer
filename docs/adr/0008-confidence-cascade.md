# ADR-0008: Confidence cascades, with failed attempts paid for

- **Status:** Accepted. Experiment complete (2026-11-12); hypothesis **not supported** under the primary cost
  model. See Outcome.
- **Date:** 2026-11-11

## Context
In v0.1, learned selectors built on cheap *pre-generation* features (image and question statistics, question
embeddings) did not beat a selector that only sees the task label. Those features mostly recover *task
identity*. They cannot see whether *this* compression damaged *this* request.

The model's confidence under a given compression is measured on the compressed computation itself, and
it comes for free: the logits already exist.

## Decision
1. **Signals** (`uniinfer.signals`), all from greedy-decoding logits and oriented so higher = more confident:
   first-token probability, first-token margin (p1 − p2), negative first-token entropy, and answer mean and
   min token log-prob. The sweep stores all of them for every (request, config).
2. **Policy** (`uniinfer.controller.cascade`): a ladder (c₁ → c₂ [→ c₃]) with thresholds. Run c₁, accept
   if confidence ≥ t₁, otherwise escalate. The ladder and thresholds are found by exhaustive search on
   training folds under the average-cost budget. Static configs are included as one-stage ladders, so a
   cascade can never be worse than static *on training data*.
3. **Cost model (primary): `sum`.** An escalated request pays for **every** attempted stage, i.e.
   C(c₁) + C(c₂). This is conservative: it assumes the cheap attempt's memory is held for its whole
   duration, and that nothing is reused across stages.
   **Secondary: `max`.** Memory freed before the retry, so cost = the most expensive stage. Reported only
   as a sensitivity check, never as the headline.
4. **Evaluation:** same 5-fold × 3-seed request-level CV as v0.1 (ADR-0006). Compare at **realized**
   held-out cost against static, per-task, task-aware and learned-T1 selectors. Report AUROC of each
   signal for "this compressed config keeps the uncompressed answer".

## Alternatives considered
- **Use confidence as a selector feature.** Impossible before running a config, which is the whole
  point. A cascade is the natural way to use post-hoc confidence.
- **Reuse the prefill when escalating.** If only the KV ratio changes, the uncompressed prefill cache
  could be re-compressed instead of re-run. That would make escalation cheaper, but only if the full cache
  is kept around, which defeats the memory saving. Left for the serving work (v0.3).
- **Learned stopping rule** (a classifier on several signals). A natural next step if single-signal
  thresholds already help.

## Consequences
- Latency is not free: escalated requests pay for two prefills. With the `sum` cost model the *memory*
  accounting is honest. Latency accounting arrives with dedicated-GPU measurements (v0.3).
- Falsifiable: if no cascade beats task-aware allocation at matched realized cost, confidence is not the
  missing per-request signal, and we report that.

## Outcome (2026-11-12)
Full numbers are in [`experiments/cascade/README.md`](../../experiments/cascade/README.md).
- **Signal:** answer min log-prob detects compression damage with AUROC 0.845. First-token signals are at
  chance on DocVQA.
- **Primary (`sum` cost):** no cascade beats task-aware allocation through confidence. A per-task cascade
  beats it at 15% memory, but the *shuffled-confidence* control does equally well, so the gain comes from
  mixing configs, not from the signal. By this ADR's own falsification rule, confidence is **not** the
  missing per-request signal *when retries are paid in full*.
- **Secondary (`max` cost):** real confidence beats the shuffled control by +1.4 to +3.2 points at 3 of 4
  budgets (paired bootstrap CIs exclude 0).
- **Added during the experiment** (exploratory, not pre-registered): per-task cascades under a shared
  budget (`fit_group_cascades`) and the shuffled-confidence control. Every later cascade claim must
  beat the shuffled control, not just static.
- **Consequence:** cheap escalation (reusing or re-compressing the prefill cache, or freeing memory before
  a retry) becomes a v0.3 serving item. The cascade's value depends on it.
