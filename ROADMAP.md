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

- ✅ **Confidence-driven cascade** ([results](experiments/cascade/README.md)). Answer log-prob detects damage
  (AUROC 0.85), but when retries are paid in full it adds nothing over a shuffled control. It helps
  (+1.4 to +3.2 pts) only when memory is freed before the retry, so the bottleneck is **escalation cost**.
- 📋 **Learned stopping rule:** combine several confidence signals with task identity; must beat the
  shuffled-confidence control.
- 📋 **Early-exit probe.** Run the first k layers on the *pruned* sequence and read text→image attention
  mass as a fragility signal (cost ≪ full prefill, unlike T3).
- 📋 **OCR-density and fine-detail signals.** Text-region density, edge energy and resolution vs question type.
- 📋 **Direct "cheapest safe config" classification** instead of score regression; calibration analysis.
- 📋 **Strong compression baselines:** FastV, SparseVLM, VisionZip (visual); SnapKV, PyramidKV, Ada-KV (KV).
  Re-run the oracle with them. Does the headroom survive better operators?
- 📋 **Scale and generalize:** 500+ requests per dataset, plus ChartQA, InfoVQA and MathVista; second VLM
  (LLaVA-OneVision or InternVL); a 7B model.
- ✅ **Statistics:** bootstrap CIs in `analyze.py` (#4); paired tests vs task-aware and a shuffled control for cascades.

## v0.3: Make it real in serving 📋

The memory saving matters when many requests share a KV pool. Show it end to end.

- 📋 **Cheap escalation** (motivated by v0.2): keep the uncompressed prefill KV in a paged pool and re-compress
  it instead of re-running prefill; free the cheap attempt's pages before the retry
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
- Add a dataset loader (InfoVQA) with its official metric in `uniinfer/metrics.py` (ChartQA ✅ #3)
- Plot per-category MMBench sensitivity from the committed JSONL
- ~~Bootstrap confidence intervals in `analyze.py`~~ ✅ #4
- A notebook showing which DocVQA questions are fragile and why

## Open research questions

1. Is fragility a property of the **image**, the **question**, or their **interaction**?
2. Do visual-token pruning and KV eviction interact? (Is the best pair different from the best of each?)
3. How much of the oracle gap is **irreducible** without running the full model?
4. Do allocation policies transfer across models of different sizes?
