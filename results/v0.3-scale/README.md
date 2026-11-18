# Results: SnapKV at scale, with ChartQA (v0.3)

2000 requests: TextVQA, DocVQA, MMBench and ChartQA, 500 each (shuffle seed 0, so the first 150 per task of
the first three are the v0.1/v0.2 requests). Write-up: [`experiments/scale/`](../../experiments/scale/README.md).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | 2000 requests × 13 configs: all visual tokens, uncompressed plus SnapKV at 50, 25, 15, 10, 9–2% KV, with confidence signals |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `scale_analysis.txt` | Output of `python experiments/scale/analyze.py`: per-task curves, headroom with CIs, held-out allocation and cascades with paired CIs |
| `scale_results.json` | Machine-readable version (input to `experiments/scale/figures.py`) |
| `check_b0.1501.*` | Same analysis at B = 0.1501 (the 15% rounding caveat) |
| `check_3tasks.*` | Without ChartQA (1500 requests) |
| `check_v02ids.*` | Only the 450 requests shared with v0.2 |

Hardware: one NVIDIA RTX A5000, about 1.5 hours. `--check` printed **EQUIVALENCE PASS** and `--check-attn`
printed **ATTENTION PASS** (worst relative error 0.0117) before the sweep. No request hit an error or OOM.
On the shared requests, all 2250 predictions also made on an A5000 in v0.2 are identical; 96.3% of those from
the RTX 3090 fine sweep are.
