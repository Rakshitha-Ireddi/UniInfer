# Results: PyramidKV vs SnapKV (v0.2 baselines)

Same 450 requests as v0.1/v0.2. Write-up: [`experiments/baselines/`](../../experiments/baselines/README.md#5-pyramidkv-is-much-worse-than-snapkv-on-this-vlm).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | 450 requests × 20 configs: visual {full, attn} × {0.75, 0.5, 0.25}, PyramidKV × {1.0, 0.25, 0.15, 0.1, 0.05}, with confidence signals; `kv_len` is the KV length averaged over layers |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `pyramid_analysis.txt` | Output of `python experiments/baselines/pyramid.py`: paired PyramidKV − SnapKV per config, with per-dataset differences |
| `pyramid_results.json` | Machine-readable version (input to `make_figures.py`) |

Hardware: one NVIDIA RTX 3090 (the SnapKV sweeps ran on RTX A5000s). `--check` printed
**EQUIVALENCE PASS** and `--check-attn` printed **ATTENTION PASS** before the sweep. On the 1800 configs this
sweep shares with the A5000 sweeps (all `kv@1.00` configs), 97.6% of predictions are identical and the mean
score differs by +0.002.
