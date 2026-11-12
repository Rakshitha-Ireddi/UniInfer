# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `uniinfer.stats`: stratified request bootstrap (`bootstrap_ci`, `paired_diff`) and replicate-level
  re-selection of the best static config and the oracle (`best_static_replicates`, `oracle_replicates`).
  2000 replicates in about 2 s via resampling-count matrices (#4).
- `experiments/oracle/analyze.py`: 95% CIs for static / per-task / oracle and the oracle-static gap
  (`--n-boot`, `--seed`).

### Findings
- The oracle-static gap is significant at every budget (e.g. +0.217 [0.180, 0.252] at 25% memory).
  Per-task static configs are **not** significantly better than one static config (CIs overlap).

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
