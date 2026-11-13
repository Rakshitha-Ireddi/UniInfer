# Results: v0.2 strong baselines (70-config grid)

Raw results are committed, so every number can be checked on a CPU (`python experiments/baselines/analyze.py`).
The headline analysis merges this sweep with the low-ratio SnapKV sweep in
[`../v0.2-baselines-low/`](../v0.2-baselines-low/README.md). Write-up: [`experiments/baselines/`](../../experiments/baselines/README.md).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | Same 450 requests as v0.1 × 70 configs: visual {full, uniform, dissim, attn} × {0.75, 0.5, 0.25}, KV {L2-norm, SnapKV} × {0.75, 0.5, 0.25} |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `baselines_analysis.txt` | Output of `analyze.py` on this sweep alone (budgets 50%–15%) |
| `baselines_results.json` | Machine-readable version |

Hardware: one NVIDIA RTX A5000, the same GPU model as v0.1. Before the sweep, `sweep.py --check` printed
**EQUIVALENCE PASS** and `sweep.py --check-attn` printed **ATTENTION PASS** (worst relative error 0.0117).
All 12600 predictions shared with the v0.1 sweep (the 28 L2-norm configs) are identical.
