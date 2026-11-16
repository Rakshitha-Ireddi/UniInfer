# Results: v0.2 strong baselines, low SnapKV ratios

> Superseded as the headline by [`../v0.2-baselines-fine/`](../v0.2-baselines-fine/README.md), which adds
> 1-point SnapKV steps. Kept because the coarse grid shows how much grid spacing alone moves the
> task-aware results (e.g. +0.089 at 7.5% here vs +0.026 on the fine grid).

Extends [`../v0.2-baselines/`](../v0.2-baselines/README.md) to tight budgets. Write-up and figures:
[`experiments/baselines/`](../../experiments/baselines/README.md).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | Same 450 requests × 40 configs: visual {full, uniform, dissim, attn} × {0.75, 0.5, 0.25}, SnapKV × {1.0, 0.15, 0.1, 0.05} |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `baselines_analysis.txt` | Output of the **merged** analysis: 100 configs, budgets 50% down to 5% |
| `baselines_results.json` | Machine-readable version |
| `cascade_snapkv.txt`, `.json` | Confidence cascades on this 100-config grid (15% to 5%) |

Reproduce on a CPU in about two minutes:

```bash
python experiments/baselines/analyze.py --extra results/v0.2-baselines-low/oracle_sweep.jsonl \
    --out results/v0.2-baselines-low/baselines_results.json
```

The merge appends this sweep's configs to the 70-config sweep and drops duplicates (every `kv@1.00` config
is already there). The merged file `merged_sweep.jsonl` is rebuilt on every run and is not committed.

Hardware: one NVIDIA RTX A5000. Both gates passed before the sweep. All 4500 predictions it shares with
the 70-config sweep are identical.
