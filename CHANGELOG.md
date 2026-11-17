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
- Fine SnapKV sweep (2–9% in 1-point steps; `results/v0.2-baselines-fine`, 149-config merged analysis).
  `analyze.py --extra` takes several sweeps, `--budgets` sets the budget list.
- `experiments/baselines/cascade_snapkv.py`: the v0.2 cascade study (real vs shuffled confidence, both cost
  models, paired tests) on the SnapKV grid at tight budgets, CPU only. New figure `cascade_snapkv_*.png`.
- PyramidKV (`compress_kv_pyramid`, `pyramid_budgets`, `kv_method="pyramid"`): SnapKV selection with a
  per-layer budget pyramid (β = 20). `kv_tokens` counts KV cost as the length averaged over layers
  (unchanged for every other operator).
- `experiments/baselines/learned_snapkv.py`: learned T1–T3 selectors on the SnapKV grid, and a within-task
  "needs almost no KV" detector.
- `experiments/baselines/pyramid.py`: paired PyramidKV-vs-SnapKV comparison; figure `pyramid_vs_snapkv_*.png`.
- `uniinfer.controller.group_frontiers` and `fit_group_cascades(..., fronts=)`: reuse per-task cascade
  frontiers across budgets (identical results; 387 s → 242 s on the 100-config run).

### Verified
- `--check-attn`: ATTENTION PASS (worst relative error 0.0117 over 36 layers × 6 samples, RTX A5000).
- `--check`: EQUIVALENCE PASS with the modified prefill. The v0.1 analysis reproduces byte-for-byte.
- All baseline sweeps (RTX A5000) reproduce every shared prediction: 12600/12600 with v0.1 and every
  overlap between the new sweeps.
- A flat pyramid (β = 1) reproduces SnapKV exactly (unit test). Decoding with per-layer KV lengths runs
  without errors on the GPU.

### Findings
- SnapKV keeps 0.850 of 0.855 at 25% KV (L2-norm: 0.590). With it, the oracle-static gap is +0.005 at 25%
  memory (was +0.217) and +0.026 [+0.015, +0.061] at 15% (was +0.316). Most of the v0.1 headroom came from
  the weak KV operator.
- SnapKV has a cliff: DocVQA falls from 0.70 to 0.43 between 5% and 4% KV; MMBench is unchanged even at 2%.
  The headroom comes back around the cliff: +0.176 [+0.146, +0.210] at 5%, +0.344 [+0.282, +0.366] at 4%.
- Held-out task-aware allocation is about 1.3 points worse than one static SnapKV config at ≥25% memory and
  +0.102 [+0.076, +0.127] better at 4% (MMBench moved to about 2% KV). Gains a coarse grid showed at 7.5–10%
  mostly vanish with 1-point steps (+0.089 → +0.026 at 7.5%, +0.024 → +0.000 at 10%).
- PyramidKV (β = 20) is worse than SnapKV at equal memory: −0.058 at 25% up to −0.168 at 10% (paired CIs
  exclude 0; DocVQA −0.397 at 5%); significantly worse in 13 of 16 matched configs. Results in
  `results/v0.2-pyramid` (RTX 3090; 97.6% of shared predictions match the A5000 sweeps).
- Request features still do not beat the task label on the SnapKV grid (T2/T3: +0.035 at 5%, −0.03 at 4%,
  −0.02 at 3%), and cannot detect insensitive requests within a task (AUROC 0.43–0.66). The answer's own
  confidence after a cheap attempt can (0.83).
- Cascades on the SnapKV grid: answer min log-prob still detects damage (AUROC 0.83). With memory freed
  before a retry, real confidence beats the shuffled control at 15%, 7.5% and 5% (+0.021, +0.032, +0.037;
  Holm-corrected) and reaches 0.728 at 5% vs 0.685 static. With retries paid in full, the effect is mixed.
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
