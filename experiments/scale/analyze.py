"""v0.3 scale-up: do the SnapKV findings hold on 4x more requests and a new task (ChartQA)? CPU only.

Input: a KV-only sweep (all visual tokens; SnapKV at 100% down to 2%), e.g.
    python experiments/oracle/sweep.py --n 500 --datasets textvqa,docvqa,mmbench,chartqa --v-ratios 1.0 \
        --kv-methods snapkv --kv-ratios 1.0,0.5,0.25,0.15,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02 \
        --out results/v0.3-scale/oracle_sweep.jsonl
Then:
    python experiments/scale/analyze.py            # results/v0.3-scale/scale_analysis.txt and .json

Questions (all fixed before seeing the data, mirroring experiments/baselines):
  1. SnapKV's score-vs-KV curve per task: where is the cliff?
  2. Oracle headroom over the best static SnapKV ratio, with stratified bootstrap CIs.
  3. Held-out: task-aware allocation vs static (paired CIs).
  4. Held-out: confidence cascades (answer min log-prob, memory freed before retry) vs task-aware and vs a
     shuffled-confidence control (paired CIs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
sys.path.insert(0, str(HERE.parent / "cascade"))
sys.path.insert(0, str(HERE.parent / "baselines"))
from cascade_snapkv import cv_group_cascade  # noqa: E402
from common import REPO, config_descriptors, load_sweep, oracle_score  # noqa: E402
from predictor import evaluate  # noqa: E402
from run_cascade import confidence_matrices, discrimination, shuffle_within  # noqa: E402

from uniinfer.controller import best_static_config  # noqa: E402
from uniinfer.stats import (  # noqa: E402,E501
    best_static_replicates,
    bootstrap_weights,
    oracle_replicates,
    paired_diff,
    percentile_ci,
)

RES = REPO / "results" / "v0.3-scale"
BUDGETS = [0.25, 0.15, 0.10, 0.075, 0.05, 0.04, 0.03]
SIGNAL = "answer min logprob"


def kv_only(recs, keys, S, C):
    """Keep uncompressed + SnapKV configs with all visual tokens (lets this run on mixed sweeps too)."""
    cfgs = recs[0]["configs"]
    cols = [i for i, c in enumerate(cfgs) if c["v_method"] == "full" and c["v_ratio"] >= 1.0
            and (c["kv_ratio"] >= 1.0 or c.get("kv_method") == "snapkv")]
    sub = [dict(r, configs=[r["configs"][i] for i in cols]) for r in recs]
    return sub, [keys[i] for i in cols], S[:, cols], C[:, cols]


def curves(recs, S, ds, names):
    Sc = np.minimum(S, S[:, :1])
    ratios = [c["kv_ratio"] for c in recs[0]["configs"]]
    print("\n=== 1. SnapKV on its own: luck-free score per task (all visual tokens) ===")
    print(f"  {'KV kept':>8s} {'all':>6s} " + " ".join(f"{d:>8s}" for d in names))
    out = {}
    for j in np.argsort(ratios)[::-1]:
        row = {"all": float(Sc[:, j].mean()), **{d: float(Sc[ds == d, j].mean()) for d in names}}
        out[f"{ratios[j]:.2f}"] = row
        print(f"  {ratios[j]:8.2f} {row['all']:6.3f} " + " ".join(f"{row[d]:8.3f}" for d in names))
    return out


def headroom(S, C, ds, keys, budgets, n_boot, seed):
    Sc = np.minimum(S, S[:, :1])
    W = bootstrap_weights(len(S), n_boot, seed, groups=ds)
    print(f"\n=== 2. Oracle headroom over the best static SnapKV ratio (luck-free; 95% CI, {n_boot} reps) ===")
    out = {}
    for B in budgets:
        if C.mean(0).min() > B + 1e-9:
            print(f"  B={B:5.3f}  infeasible")
            out[str(B)] = dict(infeasible=True)
            continue
        j = best_static_config(Sc, C, B)
        st, orc = float(Sc[:, j].mean()), float(oracle_score(Sc, C, B))
        st_r, or_r = best_static_replicates(Sc, C, B, W), oracle_replicates(Sc, C, B, W)
        (gl, gh) = percentile_ci(or_r - st_r)
        out[str(B)] = dict(static=st, oracle=orc, gap=orc - st, gap_ci=[gl, gh], static_config=keys[j])
        print(f"  B={B:5.3f}  static {st:.3f} ({keys[j]})  oracle {orc:.3f}  "
              f"gap {orc - st:+.3f} [{gl:+.3f}, {gh:+.3f}]")
    return out


def held_out(recs, S, C, ds, budgets, folds, seeds):
    names = np.unique(ds)
    onehot = (ds[:, None] == names[None]).astype(np.float32)
    desc = config_descriptors(recs)
    sel = [evaluate(S, C, ds, onehot, desc, folds, sd, budgets=budgets) for sd in range(seeds)]
    Q = confidence_matrices(recs)[SIGNAL]
    cas = [cv_group_cascade(S, C, Q, ds, folds, sd, budgets, "max") for sd in range(seeds)]
    shuf = [cv_group_cascade(S, C, shuffle_within(Q, ds, 100 + sd), ds, folds, sd, budgets, "max")
            for sd in range(seeds)]

    def per_req(runs, B, key=None):
        rr = [r[key][B] if key else r[B] for r in runs]
        return np.mean([x[0] for x in rr], 0), np.mean([x[1] for x in rr], 0)

    rows = {"static": lambda B: per_req(sel, B, "static"), "per-task static": lambda B: per_req(sel, B, "per-task"),
            "task-aware": lambda B: per_req(sel, B, "selector"), "cascade (memory freed)": lambda B: per_req(cas, B),
            "  same, shuffled confidence": lambda B: per_req(shuf, B)}
    print(f"\n=== 3-4. Held-out score [realized cost] ({folds}-fold x {seeds} seeds, raw scores) ===")
    print(f"  {'method':28s}" + "".join(f"{'B=' + str(B):>15s}" for B in budgets))
    table = {}
    for name, f in rows.items():
        vals = {B: f(B) for B in budgets}
        table[name.strip()] = {str(B): dict(score=float(v[0].mean()), cost=float(v[1].mean())) for B, v in vals.items()}
        print(f"  {name:28s}" + "".join(f"{v[0].mean():.3f} [{v[1].mean():.3f}]".rjust(15) for v in vals.values()))
    comps = {"task-aware - static": ("task-aware", "static"),
             "cascade - task-aware": ("cascade (memory freed)", "task-aware"),
             "cascade - shuffled": ("cascade (memory freed)", "  same, shuffled confidence")}
    paired = {}
    print("\n  Paired differences (95% CI, stratified request bootstrap):")
    for title, (x, y) in comps.items():
        paired[title] = {}
        cells = []
        for B in budgets:
            d = paired_diff(rows[x](B)[0], rows[y](B)[0], groups=ds)
            paired[title][str(B)] = d
            cells.append(f"{d['diff']:+.3f} [{d['low']:+.3f},{d['high']:+.3f}]")
        print(f"  {title:22s} " + "  ".join(f"B={B}: {c}" for B, c in zip(budgets, cells)))
    return table, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(RES / "oracle_sweep.jsonl"))
    ap.add_argument("--out", default=str(RES / "scale_results.json"))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--datasets", default=None, help="comma-separated subset of tasks (default: all)")
    ap.add_argument("--ids-from", default=None, help="keep only requests whose id is in this sweep jsonl")
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",")]
    recs, keys, S, C, ds = load_sweep(a.sweep)
    recs, keys, S, C = kv_only(recs, keys, S, C)
    m = np.ones(len(S), bool)
    if a.datasets:
        m &= np.isin(ds, a.datasets.split(","))
    if a.ids_from:
        ids = {json.loads(line)["id"] for line in open(a.ids_from)}
        m &= np.array([r["id"] in ids for r in recs])
    recs, S, C, ds = [r for r, k in zip(recs, m) if k], S[m], C[m], ds[m]
    names = list(np.unique(ds))
    counts = ", ".join(f"{d}={int((ds == d).sum())}" for d in names)
    print(f"requests: {len(S)} ({counts}), configs: {len(keys)}")
    print("uncompressed: " + "  ".join(f"{d} {S[ds == d, 0].mean():.3f}" for d in names)
          + f"  all {S[:, 0].mean():.3f}")
    auc = discrimination(S, {SIGNAL: confidence_matrices(recs)[SIGNAL]}, ds)[SIGNAL]
    print(f"AUROC of {SIGNAL} for 'config keeps the answer': all {auc['all']:.3f}  "
          + "  ".join(f"{d} {auc[d]:.3f}" for d in names))
    res = dict(n=len(S), per_task_n={d: int((ds == d).sum()) for d in names}, configs=keys, auroc=auc,
               curves=curves(recs, S, ds, names), headroom=headroom(S, C, ds, keys, budgets, a.n_boot, 0))
    res["held_out"], res["paired"] = held_out(recs, S, C, ds, budgets, a.folds, a.seeds)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
