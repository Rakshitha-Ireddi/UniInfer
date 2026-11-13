# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Attention-guided operators on SDPA via attention reconstruction (ADR-0009): `WindowQueryCapture`,
  `window_attention`, `compress_kv_snapkv` (SnapKV), and `select_visual_tokens(..., "attn")` (FastV-style).
- `QwenVLRunner.check_attention` and `sweep.py --check-attn`: correctness gate for the reconstruction
  (rebuilt attention output vs the model's own, every layer).
- `sweep.py --v-methods/--kv-methods`; config keys name non-default KV methods (`attn@0.50|snapkv@0.25`).
- `experiments/baselines`: operator strength, headroom per operator family with CIs, held-out allocation
  with paired CIs (task-aware minus static), infeasible-budget handling, `--extra` to merge a second sweep
  over the same requests, and three figure pairs. Results in `results/v0.2-baselines` (70 configs) and
  `results/v0.2-baselines-low` (30 low-ratio SnapKV configs; merged 100-config analysis). ADR-0009.
- `sweep.py --v-ratios/--kv-ratios`; `predictor.evaluate(..., budgets=)`.

### Verified
- `--check-attn`: ATTENTION PASS (worst relative error 0.0117 over 36 layers × 6 samples, RTX A5000).
- `--check`: EQUIVALENCE PASS with the modified prefill. The v0.1 analysis reproduces byte-for-byte.
- Both baseline sweeps (RTX A5000) reproduce every shared prediction: 12600/12600 with v0.1 and 4500/4500
  with each other.

### Findings
- SnapKV keeps 0.850 of 0.855 at 25% KV (L2-norm: 0.590). With it, the oracle-static gap is +0.005 at 25%
  memory (was +0.217) and +0.026 [+0.015, +0.061] at 15% (was +0.316). Most of the v0.1 headroom came from
  the weak KV operator.
- Headroom remains at tight budgets: +0.085 [+0.038, +0.105] at 10% and +0.176 [+0.146, +0.209] at 5%.
  SnapKV breaks on TextVQA/DocVQA below 10%; MMBench is unchanged even at 5%.
- Held-out task-aware allocation is 1.3–1.6 points worse than one static SnapKV config at ≥25% memory and
  better at 10% (+0.024 [+0.011, +0.039]) and 7.5% (+0.089, partly a grid effect).
- FastV-style (pre-prefill) visual pruning helps only at 75% keep.

## [0.2.0] - 2026-11-12

### Added
- `uniinfer.stats`: stratified request bootstrap (`bootstrap_ci`, `paired_diff`) and replicate-level
  re-selection of the best static config and the oracle (`best_static_replicates`, `oracle_replicates`).
  2000 replicates in about 2 s via resampling-count matrices (#4).
- `experiments/oracle/analyze.py`: 95% CIs for static / per-task / oracle and the oracle-static gap
  (`--n-boot`, `--seed`).
- Confidence signals from greedy logits (`uniinfer.signals`), stored for every config by the sweep, and
  `QwenVLRunner.run_with_confidence` (#6).
- `uniinfer.controller.cascade`: `Cascade`, `simulate`, `fit_cascade` (vectorized exhaustive search),
  `cascade_frontier`, and `fit_group_cascades` / `simulate_groups` (one cascade per task under one shared
  budget, exact frontier merge) (#6).
- `experiments/cascade`: evaluation harness with AUROC, held-out comparisons, a shuffled-confidence control,
  paired bootstrap tests, and figures. Results in `results/v0.2-cascade`. ADR-0008.
- ChartQA loader (content-hash ids) and relaxed-accuracy metric (#3).

### Findings
- The oracle-static gap is significant at every budget (e.g. +0.217 [0.180, 0.252] at 25% memory).
  Per-task static configs are **not** significantly better than one static config (CIs overlap).
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
