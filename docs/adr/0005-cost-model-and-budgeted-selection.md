# ADR-0005: Cost = relative KV memory; average budget; Lagrangian selection

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
To compare "static" against "per-request" allocation we need (1) one cost per (request, config), (2) a
budget definition, and (3) a selection rule that respects the budget.

## Decision
- **Cost** `C[i,c] = kv_len(i,c) / kv_len(i, uncompressed)`, the relative KV-cache memory for request
  *i*. It combines both knobs: pruning visual tokens shortens the cache, and KV eviction shortens it further.
- **Budget** is an **average** over requests: `mean_i C[i, c_i] <= B`. This is how a server with a
  fixed KV pool behaves: some requests can take more memory if others take less.
- **Selection:** solve the Lagrangian relaxation per request, `c_i = argmax_c Ŝ[i,c] − λ·C[i,c]`.
  λ is tuned on training requests to meet B. The oracle uses true scores S; a learned selector uses
  predicted scores Ŝ (`uniinfer.controller.BudgetedSelector`).
- Report the **realized** held-out cost next to every score, and plot score against realized cost.
  Then a selector that overshoots its budget can't look better than it is.

## Alternatives considered
- **Latency as the cost.** Noisy on shared GPUs (ADR-0006), and at batch 1 the weights dominate. Latency
  is recorded and will be a second cost axis once measured on dedicated hardware.
- **Per-request hard budget** (every request ≤ B). Also reported (`oracle(each)` in `analyze.py`), but it
  hides the reallocation effect we want to measure.

## Consequences
- At batch 1 and short context, KV memory is small next to the 7.6 GB of weights, so *peak VRAM* barely
  moves (see the smoke test). The memory argument applies to serving many concurrent requests or long
  contexts, which is what v0.3 must demonstrate end to end.
