# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- CPU percentile bootstrap confidence intervals for means (`uniinfer.stats.bootstrap_ci`) and 95%
  request-level intervals for luck-free static, per-task and average-budget oracle scores. Bootstrap
  replicates reselect configurations; cached oracle decisions avoid repeated lambda-grid searches.

### Fixed
- Preserve sampled-row summation order for static and per-task bootstrap selection at cost ties;
  display luck-free scores and confidence intervals together in each table cell.

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
