# Results: v0.2 confidence cascades

Raw results are committed, so every number in the README can be checked on a CPU in about two minutes
(`python experiments/cascade/run_cascade.py`).

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | Same 450 requests × 28 configs as v0.1, plus per-config confidence: `first_prob`, `first_margin`, `first_entropy`, `first_logprob`, `ans_mean_logprob`, `ans_min_logprob`, `ans_sum_logprob` |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `cascade_analysis.txt` | Full output of `experiments/cascade/run_cascade.py` |
| `cascade_results.json` | Machine-readable AUROC, held-out table and paired tests (input to `make_figures.py`) |

Hardware: one NVIDIA RTX 3090 (24 GB), shared with other jobs. v0.1 ran on an RTX A5000. The correctness
gate (`sweep.py --check`) printed **EQUIVALENCE PASS** on the 3090 before the sweep.

## Cross-GPU reproducibility

Same requests, same code path for the predictions (confidence logging only reads logits that were
already computed):

| | v0.1 (A5000) | v0.2 (3090) |
|---|:-:|:-:|
| Predictions identical to v0.1 | n/a | 12050 / 12600 (95.6%) |
| Uncompressed predictions identical | n/a | 97.8% |
| Mean score over all configs | 0.6773 | 0.6790 |
| Uncompressed score (TextVQA / DocVQA / MMBench) | 0.804 / 0.912 / 0.847 | 0.811 / 0.916 / 0.847 |

BF16 matmul kernels differ between GPU models, so near-tie greedy tokens can flip. Aggregate scores
agree to within 0.007, and every v0.1 conclusion holds on the new sweep (static, task-aware and learned
rows in `cascade_analysis.txt`). We report each version on its own sweep and never mix them.
