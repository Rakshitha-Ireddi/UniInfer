# Experiment: confidence cascades (v0.2, issue #6)

**Hypothesis (pre-registered in [ADR-0008](../../docs/adr/0008-confidence-cascade.md)).** The model's own
confidence under compression is a per-request fragility signal. Answering cheaply and escalating only when
unsure beats static and task-aware allocation at equal realized memory, *even after paying for the failed
attempts*.

**Verdict: not supported as stated.** Confidence is a strong damage detector (AUROC 0.85). But once
failed attempts are paid for, it adds nothing over a shuffled-confidence control. The signal pays off
only if a retry does not double the memory bill. Details below.

## Pipeline

| Step | Command | Needs | Output (`results/v0.2-cascade/`) |
|---|---|---|---|
| 0 | `python experiments/oracle/sweep.py --check` | GPU | `EQUIVALENCE PASS` |
| 1 | `python experiments/oracle/sweep.py --n 150 --out results/v0.2-cascade/oracle_sweep.jsonl` | GPU, about 45 min on a shared 3090 | same 450 requests × 28 configs as v0.1, **plus confidence fields** |
| 2 | `python experiments/cascade/run_cascade.py` | CPU, about 2 min | `cascade_analysis.txt`, `cascade_results.json` |
| 3 | `python experiments/cascade/make_figures.py` | CPU | `docs/figures/cascade_*.png` |

## Results

All numbers are held-out (5-fold request-level CV × 3 seeds) at **realized** average KV cost.

### 1. Confidence detects compression damage

AUROC for "this compressed config keeps the uncompressed answer" (0.5 = chance):

| Signal | All | TextVQA | DocVQA | MMBench |
|---|:-:|:-:|:-:|:-:|
| answer min log-prob | **0.845** | 0.768 | **0.805** | 0.869 |
| answer mean log-prob | 0.835 | 0.780 | 0.809 | 0.872 |
| first-token −entropy | 0.625 | 0.559 | 0.502 | 0.848 |
| first-token prob | 0.597 | 0.548 | 0.493 | 0.895 |
| first-token margin | 0.587 | 0.540 | 0.494 | 0.906 |

First-token signals only work on MMBench, where the first token *is* the answer (a letter). On open-ended
answers the damage shows up later in the answer, so whole-answer log-probs are needed.

### 2. Cascades vs baselines (sum cost: failed attempts are paid for)

| Avg. KV budget | Static | Task-aware | Cascade (one ladder) | **Task-aware cascade** | Same, confidence **shuffled** | Oracle (luck-free) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 50% | 0.769 | 0.814 | 0.783 | **0.830** | 0.827 | 0.858 |
| 37.5% | 0.708 | 0.788 | 0.752 | **0.803** | 0.801 | 0.858 |
| 25% | 0.665 | 0.721 | 0.673 | 0.714 | **0.738** | 0.858 |
| 15% | 0.543 | 0.581 | 0.575 | **0.613** | 0.605 | 0.855 |

<sub>Signal: answer min log-prob (best AUROC). Std over seeds ≤ 0.011. "Task-aware cascade" fits one ladder
and threshold per task under one shared budget (`fit_group_cascades`). It was added after the one-ladder
result, so treat it as exploratory.</sub>

- **A single cascade for all tasks loses to task-aware allocation.** The fitted ladder escalates 84–95%
  of DocVQA requests and pays twice for them.
- **A per-task cascade beats task-aware allocation** at 15% memory (+0.033, 95% CI [+0.011, +0.056]), and
  narrowly at 50% (+0.017 [−0.000, +0.034]).
- **The shuffled control explains that gain.** With confidence permuted across requests within each task
  (same distribution, no per-request link), the cascade does just as well: real − shuffled = +0.004,
  +0.002, −0.024, +0.008, and every CI covers zero or lies below it. The gain comes from *mixing two
  configs per task* (a randomized point between two static configs), not from confidence.

### 3. Where confidence does help: when a retry does not double the memory bill

Same comparison under the `max` cost model (the cheap attempt's memory is freed before the retry):

| Avg. KV budget | Task-aware | Cascade, real confidence | Cascade, shuffled | Real − shuffled [95% CI] |
|:-:|:-:|:-:|:-:|:-:|
| 50% | 0.814 | **0.841** | 0.827 | **+0.014 [+0.002, +0.026]** |
| 37.5% | 0.788 | **0.820** | 0.797 | **+0.023 [+0.008, +0.039]** |
| 25% | 0.721 | **0.769** | 0.737 | **+0.032 [+0.009, +0.054]** |
| 15% | 0.581 | 0.604 | 0.615 | −0.011 [−0.031, +0.008] |

Under `max` cost the per-request signal is worth +1.4 to +3.2 points at 3 of 4 budgets, and the cascade
beats task-aware allocation by up to +4.8 points (0.769 vs 0.721 at 25%).

### Takeaway

The per-request information exists, and the model's own answer confidence captures it. **The bottleneck
is the cost of escalating, not the signal.** Making a retry cheap means re-compressing a retained cache
instead of re-running prefill, or freeing memory before the retry in a paged KV pool. That is a serving
problem, so it moves to v0.3 ([ROADMAP](../../ROADMAP.md)).

## Controls and checks
- **Negative control** (`--dry-run`): v0.1 data with *random* confidence. One-ladder cascades gain at
  most +0.021 over static (from config mixing alone). This is why the shuffled control above is needed.
- **Shuffled control:** confidence permuted within each task, refit on every fold, same seeds.
- **Paired tests:** per-request held-out differences on identical folds; dataset-stratified bootstrap.
- **Search correctness:** the vectorized search must equal `simulate()` on its output, and the group
  frontier must contain the single-group optimum (`tests/test_cascade.py`).
- **Sanity:** 95.6% of predictions are identical to v0.1 on a different GPU model; see
  [`results/v0.2-cascade/README.md`](../../results/v0.2-cascade/README.md).
