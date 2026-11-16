# Results: v0.2 strong baselines, fine SnapKV ratios (headline analysis)

Adds SnapKV at 2–9% of the KV cache in 1-point steps to the 70-config grid ([`../v0.2-baselines/`](../v0.2-baselines/README.md))
and the low-ratio grid ([`../v0.2-baselines-low/`](../v0.2-baselines-low/README.md)), for **149 configs** per
request. Write-up and figures: [`experiments/baselines/`](../../experiments/baselines/README.md).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | Same 450 requests × 56 configs: visual {full, dissim, attn} × {0.75, 0.5, 0.25}, SnapKV × {1.0, 0.09, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02}, with confidence signals |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `baselines_analysis.txt` | Merged 149-config analysis: operators, headroom with CIs, held-out allocation with paired CIs (budgets 50% to 2%) |
| `baselines_results.json` | Machine-readable version (input to `make_figures.py`) |
| `cascade_snapkv.txt` | Confidence cascades on the 149-config grid at 15% to 3%: AUROC, held-out table, paired tests vs task-aware and a shuffled control |
| `cascade_snapkv.json` | Machine-readable version |

Commands: see the pipeline in the [experiment README](../../experiments/baselines/README.md#pipeline).

Hardware: one NVIDIA RTX A5000. `--check` printed **EQUIVALENCE PASS** and `--check-attn` printed
**ATTENTION PASS** (worst relative error 0.0028) before the sweep. All predictions it shares with the other
sweeps are identical.
