# Experiment: confidence cascades (v0.2, issue #6)

**Hypothesis.** The model's own confidence under compression is a per-request fragility signal.
Answering cheaply and escalating only when unsure beats static and task-aware allocation at equal
realized memory, *even after paying for the failed attempts* ([ADR-0008](../../docs/adr/0008-confidence-cascade.md)).

## Pipeline

| Step | Command | Needs | Output (`results/v0.2-cascade/`) |
|---|---|---|---|
| 0 | `python experiments/oracle/sweep.py --check` | GPU | `EQUIVALENCE PASS` |
| 1 | `python experiments/oracle/sweep.py --n 150 --out results/v0.2-cascade/oracle_sweep.jsonl` | GPU, about 25 min | same 450 requests × 28 configs as v0.1, **plus confidence fields** |
| 2 | `python experiments/cascade/run_cascade.py` | CPU | `cascade_results.{txt,json}` |
| 3 | `python experiments/cascade/make_figures.py` | CPU | `docs/figures/cascade_*.png` |

`run_cascade.py --dry-run` runs the whole analysis on v0.1 data with **random** confidence. Cascades
must then gain nothing over static. This is the harness's negative control (verified).

## What is reported
1. **Discrimination:** AUROC of each signal for "compressed config keeps the uncompressed answer", overall
   and per dataset.
2. **Held-out quality vs realized cost** for static, per-task, task-aware, learned-T1, cascades (one per
   signal; 3-stage with the best signal; `max`-cost sensitivity) and the oracle.
3. **What was learned:** the chosen ladder, thresholds and escalation rate per dataset.
4. **Sanity:** the new sweep's predictions are compared with v0.1's (they must be identical, since
   confidence logging must not change outputs).
