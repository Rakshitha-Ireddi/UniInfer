# Experiment: cheap escalation, end to end on new requests (v0.3 groundwork)

**Question.** On the SnapKV grid, a confidence cascade beats its shuffled control only if a retry does not
pay for the failed attempt, and request features cannot replace the cheap answer's confidence (see
[`experiments/baselines/`](../baselines/README.md), sections 4 and 6). Can a real system make the retry
cheap, and does the cascade's gain hold up on requests it has never seen?

**Method** ([ADR-0010](../../docs/adr/0010-cheap-escalation.md)). Cascades vary only the SnapKV ratio (all
visual tokens kept), so both stages come from **one prefill**. The stage-2 cache is parked in pinned CPU
memory, the stage-1 cache is decoded, and on low confidence (the answer's min token log-prob) stage 1 is
freed and stage 2 is copied back. The naive alternative re-runs encode and prefill on a retry. Both return
exactly the single-config answers (`run.py --check`), so only cost and latency differ.

## Pipeline

| Step | Command | Needs | Output (`results/v0.3-escalation/`) |
|---|---|---|---|
| 1 | `python experiments/escalation/plan.py` | CPU, about 2 min | `plan.json`: per-task cascades and the static SnapKV ratio at 7.5% and 5%, fit on the 450 v0.2 requests |
| 2 | `python experiments/escalation/run.py --check` | GPU | `ESCALATION PASS` |
| 3 | `python experiments/escalation/run.py --n 100` | GPU | `runs.jsonl`: 100 **new** requests per dataset (shuffle seed 1, every v0.1/v0.2 request skipped) × {uncompressed, static, cascade + offload, cascade + re-prefill} × 2 budgets |
| 4 | `python experiments/escalation/analyze.py` | CPU | `escalation_analysis.txt`, `escalation_results.json` |

## The plan (fit on the v0.2 requests)

KV-only grid: uncompressed plus SnapKV at 75–2% (14 configs). Cross-validated on the v0.2 requests:

| Budget | Static SnapKV | Per-task cascade | Same, shuffled confidence | Cascades chosen |
|:-:|:-:|:-:|:-:|---|
| 7.5% | 0.770 (7%) | 0.802 | 0.798 | DocVQA 7% → 15%, TextVQA 7% → 15%, MMBench 2% |
| 5% | 0.685 (5%) | 0.717 | 0.697 | DocVQA 5% → 7%, TextVQA 2% → 6%, MMBench 2% |

## Results

*Pending: the GPU run is in progress.*
