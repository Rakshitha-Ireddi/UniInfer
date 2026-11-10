# Results: v0.1 oracle study

Raw results are committed so that every number in the README can be checked on a CPU.

| File | Contents |
|---|---|
| `oracle_sweep.jsonl` | One line per request: `id`, `dataset`, `answers`, `n_vis`, and `configs[]` (28 entries: `v_method`, `v_ratio`, `kv_ratio`, `pred`, `score`, `kv_len`, `seq_len`, `prefill_s`, `decode_s`) |
| `oracle_sweep_meta.json` | Model, GPU, torch version, grid, seed |
| `features.npz` | `ids`, `t1` (request stats), `t2` (question embedding), `t3` (prefill hidden state), `times` |
| `oracle_analysis.txt` | Output of `experiments/oracle/analyze.py` |
| `predictor_results.txt` / `.json` | Output of `experiments/oracle/predictor.py` |

Hardware: one NVIDIA RTX A5000 (24 GB), **shared** with other jobs, so the `*_s` timing fields are
diagnostic only. Software: torch 2.5.1+cu121, transformers 4.51.3.
