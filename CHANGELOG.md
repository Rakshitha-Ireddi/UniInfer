# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Confidence signals from greedy logits (`uniinfer.signals`), stored for every config by the sweep, and
  `QwenVLRunner.run_with_confidence` (#6).
- `uniinfer.controller.cascade`: `Cascade`, `simulate`, `fit_cascade` (vectorized exhaustive search),
  `cascade_frontier`, and `fit_group_cascades` / `simulate_groups` (one cascade per task under one shared
  budget, exact frontier merge) (#6).
- `experiments/cascade`: evaluation harness with AUROC, held-out comparisons, a shuffled-confidence control,
  paired bootstrap tests, and figures. Results in `results/v0.2-cascade`. ADR-0008.
- ChartQA loader (content-hash ids) and relaxed-accuracy metric (#3).

### Findings
- Answer-level log-prob detects compression damage (AUROC 0.845). When failed attempts are paid for, a
  cascade's gain over task-aware allocation is matched by a shuffled-confidence control. When memory is
  freed before a retry, real confidence adds +1.4 to +3.2 points (significant at 3 of 4 budgets).
- Cross-GPU check: the v0.2 sweep (RTX 3090) reproduces 95.6% of v0.1 predictions (A5000). Mean scores
  agree to within 0.007.

## [0.1.0] - 2026-11-10

### Added
- `uniinfer.runtime.QwenVLRunner`: Qwen2.5-VL runtime with explicit encode/prefill/compress/decode that
  preserves M-RoPE positions under visual-token pruning.
- `uniinfer.compression`: `select_visual_tokens` (uniform, dissim) and `compress_kv_l2norm`.
- `uniinfer.controller`: `BudgetedSelector` and Lagrangian budgeted selection.
- `uniinfer.metrics`: VQA accuracy, ANLS, multiple-choice accuracy. `uniinfer.data`: streamed loaders for
  TextVQA, DocVQA and MMBench.
- `experiments/oracle`: sweep, features, analysis, predictor and figures; results in `results/v0.1-oracle`.
- ADRs 0001–0007, CI (ruff + pytest), contribution guidelines.

### Verified
- GPU correctness gate `sweep.py --check`: **EQUIVALENCE PASS** (6/6 samples, RTX A5000).
- Refactored sweep reproduces the committed results exactly (336/336 configs on 12 re-run requests).
- 22 unit tests pass.
