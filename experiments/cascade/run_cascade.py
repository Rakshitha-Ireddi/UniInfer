"""v0.2 experiment: confidence cascades vs static / task-aware / learned selectors / oracle.

Inputs (results/v0.2-cascade/): oracle_sweep.jsonl from `experiments/oracle/sweep.py` with confidence fields.
Request features are reused from results/v0.1-oracle/features.npz (same seed, so the same 450 requests).

    python experiments/cascade/run_cascade.py              # full evaluation (CPU, several minutes)
    python experiments/cascade/run_cascade.py --dry-run    # pipeline test on v0.1 data + random confidence
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from common import BUDGETS, REPO, load_sweep, oracle_score  # noqa: E402
from predictor import evaluate as evaluate_selectors  # noqa: E402

from uniinfer.controller import fit_cascade, fit_group_cascades, simulate, simulate_groups  # noqa: E402

V1 = REPO / "results" / "v0.1-oracle"
V2 = REPO / "results" / "v0.2-cascade"

# Each signal is oriented so that HIGHER = MORE confident.
SIGNALS = {
    "first-token prob": lambda c: c["first_prob"],
    "first-token margin": lambda c: c["first_margin"],
    "first-token -entropy": lambda c: -c["first_entropy"],
    "answer mean logprob": lambda c: c["ans_mean_logprob"],
    "answer min logprob": lambda c: c["ans_min_logprob"],
}


def confidence_matrices(recs, rng=None):
    if rng is not None:  # dry run: random confidence with the right shape
        N, K = len(recs), len(recs[0]["configs"])
        return {name: rng.random((N, K)) for name in SIGNALS}
    return {name: np.array([[f(c) for c in r["configs"]] for r in recs], float) for name, f in SIGNALS.items()}


def discrimination(S, Qs, ds):
    """AUROC of each signal for 'this compressed config keeps the uncompressed score' (pooled over configs)."""
    keep = (S[:, 1:] >= S[:, :1] - 1e-9)
    out = {}
    for name, Q in Qs.items():
        row = {"all": roc_auc_score(keep.ravel(), Q[:, 1:].ravel())}
        for d in np.unique(ds):
            m = ds == d
            y = keep[m].ravel()
            row[d] = roc_auc_score(y, Q[m, 1:].ravel()) if 0 < y.mean() < 1 else float("nan")
        out[name] = row
    return out


def cv_cascade(S, C, Q, folds, seed, budgets, **kw):
    """Held-out per-request (score, cost) for a cascade refit on every training fold."""
    N = len(S)
    res = {B: [np.zeros(N), np.zeros(N), []] for B in budgets}
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(S):
        for B in budgets:
            cas, _, _ = fit_cascade(S[tr], C[tr], Q[tr], B, **kw)
            sc, co, _ = simulate(cas, S[te], C[te], Q[te], kw.get("cost_mode", "sum"))
            res[B][0][te], res[B][1][te] = sc, co
            res[B][2].append(cas)
    return res


def shuffle_within(Q, groups, seed):
    """Permute requests' confidence rows within each group (control: same distribution, no per-request link)."""
    rng = np.random.default_rng(seed)
    out = Q.copy()
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        out[ix] = Q[rng.permutation(ix)]
    return out


def cv_group_cascade(S, C, Q, groups, folds, seed, budgets, **kw):
    """Held-out per-request (score, cost): one cascade per task, ONE shared budget (post-hoc, see README)."""
    N = len(S)
    res = {B: [np.zeros(N), np.zeros(N), []] for B in budgets}
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(S):
        for B in budgets:
            per = fit_group_cascades(S[tr], C[tr], Q[tr], groups[tr], B, **kw)
            sc, co, _ = simulate_groups(per, S[te], C[te], Q[te], groups[te], kw.get("cost_mode", "sum"))
            res[B][0][te], res[B][1][te] = sc, co
            res[B][2].append(per)
    return res


def summarize(runs, B):
    s = [r[B][0].mean() for r in runs]
    c = [r[B][1].mean() for r in runs]
    return float(np.mean(s)), float(np.std(s)), float(np.mean(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t_start = time.time()

    sweep = V1 / "oracle_sweep.jsonl" if a.dry_run else V2 / "oracle_sweep.jsonl"
    recs, keys, S, C, ds = load_sweep(sweep)
    Qs = confidence_matrices(recs, np.random.default_rng(0) if a.dry_run else None)
    print(f"{'DRY RUN: ' if a.dry_run else ''}N={len(S)} configs={S.shape[1]} folds={a.folds} seeds={a.seeds}")

    # ---------------------------------------------------------------- sanity: same outputs as v0.1
    if not a.dry_run:
        old = {}
        for line in open(V1 / "oracle_sweep.jsonl"):
            r = json.loads(line)
            old[r["id"]] = [c["pred"] for c in r["configs"]]
        pairs = [(p, c["pred"]) for r in recs if r["id"] in old for p, c in zip(old[r["id"]], r["configs"])]
        same = sum(p == q for p, q in pairs)
        _, _, S1, _, _ = load_sweep(V1 / "oracle_sweep.jsonl")
        print(f"sanity: {same}/{len(pairs)} predictions identical to v0.1 "
              f"({sum(r['id'] in old for r in recs)}/{len(recs)} requests matched); "
              f"mean score {S1.mean():.4f} (v0.1) vs {S.mean():.4f} (this sweep). Differences come from BF16 "
              "kernels on a different GPU model, not from confidence logging (see the README).")

    # ---------------------------------------------------------------- 1. discrimination
    auc = discrimination(S, Qs, ds)
    names = list(np.unique(ds))
    print("\n=== 1. Does confidence detect damage? AUROC for 'compressed config keeps the answer' ===")
    print(f"  {'signal':24s} {'all':>6s} " + " ".join(f"{d:>8s}" for d in names))
    for s, row in auc.items():
        print(f"  {s:24s} {row['all']:6.3f} " + " ".join(f"{row[d]:8.3f}" for d in names))
    print("  (0.5 = no information, 1.0 = perfect)")

    # ---------------------------------------------------------------- 2. held-out comparison
    F = np.load(V1 / "features.npz")
    pos = {i: k for k, i in enumerate(F["ids"])}
    have = all(r["id"] in pos for r in recs)
    from common import config_descriptors
    desc = config_descriptors(recs)
    table = {}

    base = [evaluate_selectors(S, C, ds, None, desc, a.folds, sd) for sd in range(a.seeds)]
    for m in ["static", "per-task"]:
        table[m] = {B: (np.mean([r[m][B][0].mean() for r in base]), np.std([r[m][B][0].mean() for r in base]),
                        np.mean([r[m][B][1].mean() for r in base])) for B in BUDGETS}
    onehot = (ds[:, None] == np.unique(ds)[None]).astype(np.float32)
    feats = {"task-aware": onehot}
    if have:
        order = np.array([pos[r["id"]] for r in recs])
        feats["learned T1"] = F["t1"][order]
    held = {}
    for name, X in feats.items():
        runs = [evaluate_selectors(S, C, ds, X, desc, a.folds, sd) for sd in range(a.seeds)]
        held[name] = [r["selector"] for r in runs]
        table[name] = {B: (np.mean([r["selector"][B][0].mean() for r in runs]),
                           np.std([r["selector"][B][0].mean() for r in runs]),
                           np.mean([r["selector"][B][1].mean() for r in runs])) for B in BUDGETS}
    print(f"\n[{time.time() - t_start:.0f}s] baselines done; fitting cascades...", flush=True)

    best_sig = max(auc, key=lambda s: auc[s]["all"])
    chosen = {}
    for sig, Q in Qs.items():
        runs = [cv_cascade(S, C, Q, a.folds, sd, BUDGETS) for sd in range(a.seeds)]
        table[f"cascade-2 | {sig}"] = {B: summarize(runs, B) for B in BUDGETS}
        chosen[sig] = runs[0]
        print(f"[{time.time() - t_start:.0f}s] cascade-2 {sig} done", flush=True)
    runs3 = [cv_cascade(S, C, Qs[best_sig], a.folds, sd, BUDGETS, max_stages=3, final_stage=0, n_thresholds=11)
             for sd in range(a.seeds)]
    table[f"cascade-3 | {best_sig}"] = {B: summarize(runs3, B) for B in BUDGETS}
    runs_max = [cv_cascade(S, C, Qs[best_sig], a.folds, sd, BUDGETS, cost_mode="max") for sd in range(a.seeds)]
    table[f"cascade-2 | {best_sig} | max-cost"] = {B: summarize(runs_max, B) for B in BUDGETS}
    # post-hoc follow-up (added after the task-blind result): task-aware allocation + cascades
    runs_task = [cv_group_cascade(S, C, Qs[best_sig], ds, a.folds, sd, BUDGETS) for sd in range(a.seeds)]
    table[f"task-aware cascade | {best_sig}"] = {B: summarize(runs_task, B) for B in BUDGETS}
    # Control: the same cascade search with confidence SHUFFLED across requests within each task. It keeps the
    # signal's distribution (so thresholds still split traffic) but breaks its link to the request. Any gain
    # that survives this control comes from mixing two configs, not from confidence.
    runs_shuf = [cv_group_cascade(S, C, shuffle_within(Qs[best_sig], ds, 100 + sd), ds, a.folds, sd, BUDGETS)
                 for sd in range(a.seeds)]
    table[f"task-aware cascade | {best_sig} SHUFFLED (control)"] = {B: summarize(runs_shuf, B) for B in BUDGETS}
    # Sensitivity: the same pair under the max-cost model (memory freed before a retry; ADR-0008).
    runs_task_max = [cv_group_cascade(S, C, Qs[best_sig], ds, a.folds, sd, BUDGETS, cost_mode="max")
                     for sd in range(a.seeds)]
    runs_shuf_max = [cv_group_cascade(S, C, shuffle_within(Qs[best_sig], ds, 100 + sd), ds, a.folds, sd, BUDGETS,
                                      cost_mode="max") for sd in range(a.seeds)]
    table[f"task-aware cascade | {best_sig} | max-cost"] = {B: summarize(runs_task_max, B) for B in BUDGETS}
    table[f"task-aware cascade | {best_sig} SHUFFLED | max-cost"] = {B: summarize(runs_shuf_max, B)
                                                                     for B in BUDGETS}
    print(f"[{time.time() - t_start:.0f}s] task-aware cascade + shuffled control done", flush=True)
    Sc = np.minimum(S, S[:, :1])
    table["oracle (luck-free)"] = {B: (oracle_score(Sc, C, B), 0.0, B) for B in BUDGETS}
    table["oracle (raw)"] = {B: (oracle_score(S, C, B), 0.0, B) for B in BUDGETS}

    print("\n=== 2. Held-out score, mean ± std over seeds  [realized average cost] ===")
    print("  cascade cost = SUM over attempted stages (failed attempts are paid for) unless marked max-cost")
    w = max(len(k) for k in table) + 2
    print(f"  {'method':{w}s}" + "".join(f"{'B=' + str(B):>24s}" for B in BUDGETS))
    for k, row in table.items():
        print(f"  {k:{w}s}" + "".join(f"{row[B][0]:.3f}±{row[B][1]:.3f} [{row[B][2]:.2f}]".rjust(24) for B in BUDGETS))

    # ---------------------------------------------------------------- 3. what the cascades learned
    print(f"\n=== 3. Cascades chosen on fold 1 (seed 0), signal = {best_sig} ===")
    for B in BUDGETS:
        cas = chosen[best_sig][B][2][0]
        st = " -> ".join(keys[c] for c in cas.stages)
        th = ", ".join(f"{t:.3g}" for t in cas.thresholds)
        _, _, stage = simulate(cas, S, C, Qs[best_sig])
        esc = (stage > 0).mean()
        print(f"  B={B}: {st}   thresholds [{th}]   escalation rate {esc:.0%}")
        for d in names:
            print(f"        {d:8s} escalated {(stage[ds == d] > 0).mean():.0%}")

    print("\n=== 4. Paired tests (same requests, same folds; per-request held-out difference averaged over seeds;")
    print("       95% CI from a dataset-stratified request bootstrap, 2000 replicates) ===")
    strata = [np.where(ds == d)[0] for d in names]
    comparisons = {"task-aware cascade - task-aware selector": (runs_task, held["task-aware"]),
                   "task-aware cascade - same cascade with SHUFFLED confidence": (runs_task, runs_shuf),
                   "[max-cost] task-aware cascade - SHUFFLED confidence": (runs_task_max, runs_shuf_max)}
    paired = {}
    for title, (A, Bruns) in comparisons.items():
        print(f"  {title}")
        rng = np.random.default_rng(0)
        paired[title] = {}
        for B in BUDGETS:
            d = np.mean([x[B][0] - y[B][0] for x, y in zip(A, Bruns)], 0)
            dc = np.mean([x[B][1] - y[B][1] for x, y in zip(A, Bruns)], 0)
            reps = np.array([np.concatenate([d[rng.choice(ix, len(ix))] for ix in strata]).mean()
                             for _ in range(2000)])
            lo, hi = np.percentile(reps, [2.5, 97.5])
            paired[title][str(B)] = dict(diff=float(d.mean()), low=float(lo), high=float(hi),
                                         p_le_0=float((reps <= 0).mean()), cost_diff=float(dc.mean()))
            print(f"    B={B}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  P(diff<=0)={(reps <= 0).mean():.3f}  "
                  f"cost difference {dc.mean():+.3f}")

    print(f"\n=== 5. Task-aware cascades chosen on fold 1 (seed 0), signal = {best_sig} ===")
    for B in BUDGETS:
        per = runs_task[0][B][2][0]
        _, co, stage = simulate_groups(per, S, C, Qs[best_sig], ds)
        print(f"  B={B}: realized cost {co.mean():.2f}")
        for d, cas in per.items():
            st = " -> ".join(keys[c] for c in cas.stages)
            th = ", ".join(f"{t:.3g}" for t in cas.thresholds) or "-"
            m = ds == d
            print(f"        {d:8s} {st:45s} thr [{th}]  escalated {(stage[m] > 0).mean():.0%}  cost {co[m].mean():.2f}")

    out = V1 / "_dryrun_cascade.json" if a.dry_run else V2 / "cascade_results.json"
    json.dump({"auroc": auc, "best_signal": best_sig, "paired": paired,
               "table": {k: {str(B): dict(score=v[0], std=v[1], cost=v[2]) for B, v in row.items()}
                         for k, row in table.items()}}, open(out, "w"), indent=2)
    print(f"\nwrote {out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
