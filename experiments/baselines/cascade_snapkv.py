"""Do confidence cascades help once SnapKV is in the grid? (v0.2 follow-up, CPU only)

The baseline sweeps log the same confidence signals as the cascade sweep for every config, so the v0.2
cascade study can be re-run on the 100-config grid (or more, with --extra) without new GPU time. We test
only the budgets where the oracle-static gap survives SnapKV (15% and below).

    python experiments/baselines/cascade_snapkv.py
    python experiments/baselines/cascade_snapkv.py --extra results/v0.2-baselines-low/oracle_sweep.jsonl \
        results/v0.2-baselines-fine/oracle_sweep.jsonl --out results/v0.2-baselines-fine/cascade_snapkv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
sys.path.insert(0, str(HERE.parent / "cascade"))
sys.path.insert(0, str(HERE))  # this directory first: oracle/ also has an analyze.py
from analyze import merge_sweeps  # noqa: E402
from common import REPO, config_descriptors, load_sweep, oracle_score  # noqa: E402
from predictor import evaluate  # noqa: E402
from run_cascade import confidence_matrices, discrimination, shuffle_within  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from uniinfer.controller import fit_group_cascades, group_frontiers, simulate_groups  # noqa: E402
from uniinfer.stats import paired_diff  # noqa: E402

BASE = REPO / "results" / "v0.2-baselines"
LOW = REPO / "results" / "v0.2-baselines-low"
BUDGETS = [0.15, 0.10, 0.075, 0.05]
SIGNAL = "answer min logprob"   # best signal in the v0.2 cascade study (AUROC 0.845); fixed in advance


def auroc_by_operator(recs, S, Qs, ds):
    """AUROC per KV operator, so SnapKV damage can be compared with L2-norm damage."""
    cfgs = recs[0]["configs"]
    groups = {
        "L2-norm, all ratios": [i for i, c in enumerate(cfgs) if c["kv_ratio"] < 1 and c.get("kv_method") == "l2norm"],
        "SnapKV, all ratios": [i for i, c in enumerate(cfgs) if c["kv_ratio"] < 1 and c.get("kv_method") == "snapkv"],
        "SnapKV, keep <= 15%": [i for i, c in enumerate(cfgs)
                                if c["kv_ratio"] <= 0.15 and c.get("kv_method") == "snapkv"],
    }
    out = {}
    for name, cols in groups.items():
        if not cols:
            continue
        idx = [0] + cols
        out[name] = discrimination(S[:, idx], {SIGNAL: Qs[SIGNAL][:, idx]}, ds)[SIGNAL]
        out[name]["n_configs"] = len(cols)
    return out


def cv_group_cascade(S, C, Q, groups, folds, seed, budgets, cost_mode):
    """Held-out per-request (score, cost) of per-task cascades under one shared budget. Same protocol as
    run_cascade.cv_group_cascade, but each fold's frontiers are computed once and reused for every budget."""
    N = len(S)
    res = {B: [np.zeros(N), np.zeros(N), []] for B in budgets}
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(S):
        fronts = group_frontiers(S[tr], C[tr], Q[tr], groups[tr], cost_mode=cost_mode)
        for B in budgets:
            per = fit_group_cascades(S[tr], C[tr], Q[tr], groups[tr], B, fronts=fronts)
            sc, co, _ = simulate_groups(per, S[te], C[te], Q[te], groups[te], cost_mode)
            res[B][0][te], res[B][1][te] = sc, co
            res[B][2].append(per)
    return res


def held(runs, B):
    return np.mean([r[B][0] for r in runs], 0), np.mean([r[B][1] for r in runs], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(BASE / "oracle_sweep.jsonl"))
    ap.add_argument("--extra", nargs="*", default=[str(LOW / "oracle_sweep.jsonl")])
    ap.add_argument("--out", default=str(LOW / "cascade_snapkv"), help="output prefix (.txt is stdout, .json)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",")]
    t0 = time.time()

    path = a.sweep
    for k, extra in enumerate(a.extra):
        merged = Path(a.out).with_name(f"merged_sweep{'' if k == len(a.extra) - 1 else k}.jsonl")
        merge_sweeps(path, extra, merged)
        path = str(merged)
    recs, keys, S, C, ds = load_sweep(path)
    Qs = confidence_matrices(recs)
    names = list(np.unique(ds))
    print(f"N={len(S)} configs={S.shape[1]} budgets={budgets} signal={SIGNAL!r} folds={a.folds} seeds={a.seeds}")

    # ------------------------------------------------------------- 1. does confidence still detect damage?
    auc = auroc_by_operator(recs, S, Qs, ds)
    print("\n=== 1. AUROC of answer min logprob for 'compressed config keeps the uncompressed score' ===")
    print(f"  {'configs':22s} {'n':>4s} {'all':>6s} " + " ".join(f"{d:>8s}" for d in names))
    for k, row in auc.items():
        print(f"  {k:22s} {row['n_configs']:4d} {row['all']:6.3f} " + " ".join(f"{row[d]:8.3f}" for d in names))

    # ------------------------------------------------------------- 2. held-out comparison at tight budgets
    onehot = (ds[:, None] == np.unique(ds)[None]).astype(np.float32)
    desc = config_descriptors(recs)
    sel = [evaluate(S, C, ds, onehot, desc, a.folds, sd, budgets=budgets) for sd in range(a.seeds)]
    static = [{B: r["static"][B] for B in budgets} for r in sel]
    task = [{B: r["selector"][B] for B in budgets} for r in sel]
    print(f"[{time.time() - t0:.0f}s] selectors done", flush=True)
    Q = Qs[SIGNAL]
    runs = {}
    for mode in ["sum", "max"]:
        runs[f"cascade {mode}"] = [cv_group_cascade(S, C, Q, ds, a.folds, sd, budgets, cost_mode=mode)
                                   for sd in range(a.seeds)]
        runs[f"shuffled {mode}"] = [cv_group_cascade(S, C, shuffle_within(Q, ds, 100 + sd), ds, a.folds, sd,
                                                     budgets, cost_mode=mode) for sd in range(a.seeds)]
        print(f"[{time.time() - t0:.0f}s] cascades ({mode} cost) done", flush=True)
    rows = {"static": static, "task-aware selector": task,
            "task-aware cascade (retry pays both)": runs["cascade sum"],
            "  same, SHUFFLED confidence": runs["shuffled sum"],
            "task-aware cascade (memory freed before retry)": runs["cascade max"],
            "  same, SHUFFLED confidence ": runs["shuffled max"]}
    Sc = np.minimum(S, S[:, :1])
    table = {}
    print("\n=== 2. Held-out score [realized cost], raw scores, mean over seeds ===")
    w = max(map(len, rows)) + 2
    print(f"  {'method':{w}s}" + "".join(f"{'B=' + str(B):>16s}" for B in budgets))
    for name, rr in rows.items():
        table[name.strip()] = {str(B): dict(score=float(held(rr, B)[0].mean()), cost=float(held(rr, B)[1].mean()))
                               for B in budgets}
        print(f"  {name:{w}s}" + "".join(f"{held(rr, B)[0].mean():.3f} [{held(rr, B)[1].mean():.3f}]".rjust(16)
                                         for B in budgets))
    table["oracle (luck-free)"] = {str(B): dict(score=float(oracle_score(Sc, C, B)), cost=B) for B in budgets}
    print(f"  {'oracle (luck-free)':{w}s}" + "".join(f"{oracle_score(Sc, C, B):.3f}".rjust(16) for B in budgets))

    # ------------------------------------------------------------- 3. paired tests
    print("\n=== 3. Paired differences (per-request, averaged over seeds; stratified bootstrap 95% CI) ===")
    comps = {"cascade - task-aware selector (retry pays both)": ("cascade sum", task),
             "cascade - SHUFFLED (retry pays both)": ("cascade sum", runs["shuffled sum"]),
             "cascade - task-aware selector (memory freed)": ("cascade max", task),
             "cascade - SHUFFLED (memory freed)": ("cascade max", runs["shuffled max"]),
             "task-aware selector - static": (task, static)}
    paired = {}
    for title, (x, y) in comps.items():
        x = runs[x] if isinstance(x, str) else x
        paired[title] = {}
        print(f"  {title}")
        for B in budgets:
            (sx, cx), (sy, cy) = held(x, B), held(y, B)
            d = paired_diff(sx, sy, groups=ds)
            d["cost_diff"] = float((cx - cy).mean())
            paired[title][str(B)] = d
            print(f"    B={B}: {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]  P(<=0)={d['p_le_0']:.3f}  "
                  f"cost diff {d['cost_diff']:+.3f}")

    # ------------------------------------------------------------- 4. what the cascades chose
    print("\n=== 4. Cascades on fold 1, seed 0 (memory-freed model) ===")
    for B in budgets:
        per = runs["cascade max"][0][B][2][0]
        print(f"  B={B}")
        for d, cas in per.items():
            st = " -> ".join(keys[c] for c in cas.stages)
            th = ", ".join(f"{t:.3g}" for t in cas.thresholds) or "-"
            print(f"    {d:8s} {st:50s} thr [{th}]")

    json.dump(dict(n=len(S), n_configs=S.shape[1], signal=SIGNAL, budgets=budgets, auroc=auc, table=table,
                   paired=paired), open(a.out + ".json", "w"), indent=2)
    print(f"\nwrote {a.out}.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
