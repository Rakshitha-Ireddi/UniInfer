# ADR-0002: Qwen2.5-VL-3B on TextVQA, DocVQA and MMBench, in HF Transformers

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
We need a VLM that (a) fits on a 24 GB GPU with room for many configurations, (b) produces many visual
tokens per image so the visual budget matters, and (c) is widely used, so results transfer.
We also need tasks that should differ in how much visual detail they need.

## Decision
- **Backbone:** `Qwen/Qwen2.5-VL-3B-Instruct`, BF16, SDPA attention. About 7.8 GB peak at batch 1.
- **Resolution cap:** `max_pixels = 1024·28·28`, i.e. at most about 1024 visual tokens per image.
  Without a cap, DocVQA pages produce more than 10k tokens.
- **Benchmarks:** TextVQA (scene text, VQA accuracy), DocVQA (documents, ANLS) and MMBench-dev
  (general multiple choice, accuracy). They range from "fine detail needed" to "coarse detail is enough".
- **Framework:** Hugging Face Transformers for the research prototype. vLLM integration comes after the
  algorithm is validated (roadmap v0.3).

## Alternatives considered
- **LLaVA-OneVision / InternVL as primary.** Deferred to cross-model validation (v0.2).
- **7B model.** Fits, but leaves less memory headroom on shared GPUs and makes sweeps slower.
- **vLLM from day one.** Its scheduler makes per-request KV surgery much harder to prototype.

## Consequences
- Results are for one 3B model. Generalization is an explicit v0.2 item.
- Scores use simplified official metrics (see `src/uniinfer/metrics.py`). They are comparable across our
  configs, not directly comparable to leaderboard numbers.
