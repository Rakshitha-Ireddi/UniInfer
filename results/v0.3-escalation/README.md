# Results: cheap escalation, end to end (v0.3)

Write-up: [`experiments/escalation/`](../../experiments/escalation/README.md). Design: [ADR-0010](../../docs/adr/0010-cheap-escalation.md).

| File | Contents |
|---|---|
| `plan.json` | Per-task KV-only cascades and the static SnapKV ratio at 7.5% and 5%, fit on the 450 v0.2 requests (`plan.py`) |
| `runs.jsonl` | 300 new requests (TextVQA, DocVQA, MMBench × 100; shuffle seed 1, no v0.1/v0.2 request) × {uncompressed, static, cascade with parked cache, cascade with re-prefill} × 2 budgets: answers, scores, escalation, KV tokens per stage, timings |
| `runs_meta.json` | Model, GPU, torch version, seed, the plan used |
| `escalation_analysis.txt` | Output of `analyze.py`: scores, paired cascade − static, latency, memory |
| `escalation_results.json` | Machine-readable version |
| `transfer.txt` | Output of `transfer.py`: the plan replayed on the 1050 held-out requests of `results/v0.3-scale`, against task-aware allocation and rate-matched random escalation, plus the replay-vs-GPU check |
| `transfer.json` | Machine-readable version |

Hardware: one NVIDIA RTX A5000 shared with other jobs, so timings are diagnostic. `run.py --check` printed
**ESCALATION PASS** before the run. No request hit an error or OOM. The two escalation modes gave identical
answers on every request, and the CPU replay reproduces all 348 answers and 174 escalation decisions on the
87 requests it shares with the GPU run.
