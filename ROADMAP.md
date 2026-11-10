# Roadmap

UniInfer is a research project developed in the open. Items are **plans, not promises**: each research
item states the question it answers, and results may change the plan. Want to take something on? Comment
on its issue, or open one, so we don't duplicate work.

Legend: ✅ done · 🚧 in progress · 📋 planned · 💡 open idea (needs an owner)

---

## v0.1: Is per-request allocation worth it? ✅

- ✅ Qwen2.5-VL runtime with explicit prefill/compress/decode, token-identical to `generate()`
- ✅ Visual-token selection (uniform, redundancy) and L2-norm KV eviction
- ✅ 450-request × 28-config oracle sweep over TextVQA, DocVQA and MMBench
- ✅ Budgeted selection (Lagrangian) and learned selector, request-grouped CV
- ✅ **Result:** large oracle headroom; learned selection gains +6–9 pts over static, but most of it comes from
  task-level reallocation (see README)

## v0.2: Close the oracle gap with per-request signals 🚧

The central research question: **what cheap signal tells us, before answering, that a request is fragile?**

- 🚧 **Confidence-driven cascade.** Answer with a cheap config first. If first-token entropy or margin is low,
  escalate to a larger budget. Measure the quality/cost trade-off, including the cost of the retries.
- 📋 **Early-exit probe.** Run the first k layers on the *pruned* sequence and read text→image attention
  mass as a fragility signal (cost ≪ full prefill, unlike T3).
- 📋 **OCR-density and fine-detail signals.** Text-region density, edge energy and resolution vs question type.
- 📋 **Direct "cheapest safe config" classification** instead of score regression; calibration analysis.
- 📋 **Strong compression baselines:** FastV, SparseVLM, VisionZip (visual); SnapKV, PyramidKV, Ada-KV (KV).
  Re-run the oracle with them. Does the headroom survive better operators?
- 📋 **Scale and generalize:** 500+ requests per dataset, plus ChartQA, InfoVQA and MathVista; second VLM
  (LLaVA-OneVision or InternVL); a 7B model.
- 📋 **Statistics:** bootstrap CIs on every table; paired tests vs the task-aware baseline.

## v0.3: Make it real in serving 📋

The memory saving matters when many requests share a KV pool. Show it end to end.

- 📋 Batched decode with **heterogeneous per-request KV lengths**
- 📋 Admission control: requests/sec and tail latency under a fixed KV-memory cap
- 📋 vLLM integration (paged KV plus per-request token budgets)
- 📋 Latency and energy on a dedicated GPU (prefill/decode split, TTFT/TPOT)
- 💡 Selector overhead budget: the controller must cost less than 2% of the request

## v0.4: Compression foundation 📋

Bring weight compression in as a *static* axis that the per-request controller works around (ADR-0001).

- 📋 INT8/INT4 weight quantization (RTN, AWQ, GPTQ) with packed checkpoints
- 📋 KV-cache quantization (INT8/INT4, group-wise) as an extra per-request knob
- 📋 Fused INT4 dequant-GEMM Triton kernel and roofline analysis
- 💡 Nested ("any-precision") weights, which would make per-request weight precision feasible

## v1.0: Benchmark and paper 📋

- 📋 `uniinfer-bench`: standard protocol for adaptive-compression methods (oracle, luck-free scoring,
  realized-cost reporting)
- 📋 Paper: problem formulation, oracle headroom across models/operators, per-request selector, serving results

---

## Good first issues

- Add a compression operator, for example **random** visual-token pruning (a sanity baseline) or **recent-window** KV
- Add a dataset loader (ChartQA, InfoVQA) with its official metric in `uniinfer/metrics.py`
- Plot per-category MMBench sensitivity from the committed JSONL
- Bootstrap confidence intervals in `analyze.py`
- A notebook showing which DocVQA questions are fragile and why

## Open research questions

1. Is fragility a property of the **image**, the **question**, or their **interaction**?
2. Do visual-token pruning and KV eviction interact? (Is the best pair different from the best of each?)
3. How much of the oracle gap is **irreducible** without running the full model?
4. Do allocation policies transfer across models of different sizes?
