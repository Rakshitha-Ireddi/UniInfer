# ADR-0001: Start with request-adaptive visual-token and KV budgets; defer weight quantization

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
The original plan was a 12-module framework: weight quantization (INT4/INT8/FP8, AWQ/GPTQ/SmoothQuant),
packing, Triton kernels, KV-cache compression, visual-token pruning, and a joint controller. For one
maintainer that is several months of re-implementing existing work before any original result exists.

Two technical problems also surfaced:
1. **Per-request weight precision is not practical.** Serving different weight precisions to different
   requests means either holding several weight copies (which cancels the memory saving) or re-quantizing
   per request (too slow). It also breaks batching. Nested schemes (Any-Precision LLM, Matryoshka
   Quantization) are a separate research problem.
2. **Combining compressors is not novel by itself.** Joint quantization plus token pruning, and joint KV
   plus weight compression, have already been studied.

## Decision
v0.1 asks one falsifiable question:

> Under a fixed average memory budget, does choosing the **visual-token budget** and **KV-cache budget**
> per request beat the best **single static** configuration?

Weights stay in BF16, and weight quantization moves to the roadmap (v0.4). We run the cheapest experiment
that could kill the idea (a per-request oracle) before building any controller.

## Alternatives considered
- **Build the full framework first.** Rejected: months of work, and an engineering-only outcome if the
  hypothesis fails.
- **Per-request weight precision.** Rejected for v0.1 for the reasons above.

## Consequences
- The first result arrives in days, not months.
- The per-request knobs (visual tokens, KV) are ones a server can actually change per request.
- Quantization, kernels and serving stay visible on the roadmap as contribution areas.
