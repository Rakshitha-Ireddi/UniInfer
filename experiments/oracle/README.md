# Experiment: oracle study (v0.1)

**Question.** Under a fixed average KV-memory budget, does choosing compression *per request* beat the
best single static configuration? If so, can cheap request features make that choice?

## Pipeline

| Step | Script | Needs | Output (`results/v0.1-oracle/`) |
|---|---|---|---|
| 0 | `sweep.py --check` | GPU | prints `EQUIVALENCE PASS` (manual loop matches `generate()`) |
| 1 | `sweep.py --n 150` | GPU, about 25 min | `oracle_sweep.jsonl`: 450 requests × 28 configs (pred, score, kv_len, timings) |
| 2 | `features.py` | GPU, about 3 min | `features.npz`: T1/T2/T3 request features and their overheads |
| 3 | `analyze.py` | CPU, seconds | `oracle_analysis.txt`: static vs per-task vs oracle; sensitivity; noise check |
| 4 | `predictor.py` | CPU, minutes | `predictor_results.{txt,json}`: learned selectors, 5-fold CV × 3 seeds |
| 5 | `make_figures.py` | CPU | `docs/figures/*.png` |

## Setup

- **Model:** Qwen2.5-VL-3B-Instruct, BF16, SDPA, `max_pixels = 1024·28·28`, greedy decoding.
- **Data:** 150 shuffled requests (seed 0) each from TextVQA-val, DocVQA-val and MMBench-dev (EN).
- **Grid:** visual keep {100, 75, 50, 25}% × {uniform, dissim} × KV keep {100, 75, 50, 25}% = 28 configs.
- **Cost:** KV length relative to uncompressed. **Budget:** average over requests ([ADR-0005](../../docs/adr/0005-cost-model-and-budgeted-selection.md)).
- **Protocol:** [ADR-0006](../../docs/adr/0006-evaluation-protocol.md): luck-free oracle, request-level CV,
  every baseline chosen on training folds, realized cost reported.

## Key numbers

Luck-free oracle vs baselines (`analyze.py`):

| Budget | Static | Per-task | Oracle |
|:-:|:-:|:-:|:-:|
| 50% | 0.750 | 0.761 | 0.855 |
| 25% | 0.638 | 0.667 | 0.855 |
| 15% | 0.536 | 0.559 | 0.852 |

Learned selectors (held-out, real scores) are in the main README.
