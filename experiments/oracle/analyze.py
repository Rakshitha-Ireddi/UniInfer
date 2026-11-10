"""Step 3: static vs per-task vs per-request ORACLE under equal average KV-memory budgets.

Cost of a config = kv_len / kv_len(uncompressed), i.e. relative KV-cache memory.
Runs on CPU in seconds:  python experiments/oracle/analyze.py
"""
from __future__ import annotations

import numpy as np
from common import BUDGETS, RESULTS, load_sweep, oracle_score

from uniinfer.controller import best_static_config


def per_task_score(S, C, ds, B):
    tot = 0.0
    for d in np.unique(ds):
        m = ds == d
        tot += S[m, best_static_config(S[m], C[m], B)].sum()
    return tot / len(S)


def report(S, C, ds, keys, title):
    print(f"\n=== {title} ===")
    print(f"  {'budget':>6s}  {'static':>7s}  {'per-task':>8s}  {'oracle':>7s}  {'gap':>6s}  static config")
    for B in BUDGETS:
        j = best_static_config(S, C, B)
        st, pt, orc = S[:, j].mean(), per_task_score(S, C, ds, B), oracle_score(S, C, B)
        print(f"  {B:6.3f}  {st:7.3f}  {pt:8.3f}  {orc:7.3f}  {orc - st:+6.3f}  {keys[j]}")


def main():
    recs, keys, S, C, ds = load_sweep()
    names = np.unique(ds)
    print(f"samples: {len(S)} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in names) + f"), configs: {len(keys)}")
    print("\nUncompressed score: " + ", ".join(f"{d} {S[ds == d, 0].mean():.3f}" for d in names)
          + f", all {S[:, 0].mean():.3f}")

    print("\nMean score per config (sorted by mean cost):")
    print(f"  {'config':30s} {'cost':>5s}  " + "  ".join(f"{d:>7s}" for d in names) + f"  {'all':>6s}")
    for j in np.argsort(C.mean(0)):
        per = "  ".join(f"{S[ds == d, j].mean():7.3f}" for d in names)
        print(f"  {keys[j]:30s} {C[:, j].mean():5.2f}  {per}  {S[:, j].mean():6.3f}")

    report(S, C, ds, keys, "Raw scores")
    Sc = np.minimum(S, S[:, :1])  # no credit for a compressed config beating the uncompressed model
    report(Sc, C, ds, keys, "Luck-free scores (capped at the uncompressed score). Use this for claims.")

    print("\n=== Sensitivity: cheapest config that keeps the uncompressed score ===")
    for d in names:
        idx = np.where(ds == d)[0]
        cheapest = np.array([C[i, S[i] >= S[i, 0] - 1e-9].min() for i in idx])
        print(f"  {d:8s} median cost {np.median(cheapest):.2f}   "
              + "   ".join(f"<= {b:.2f}: {(cheapest <= b + 1e-9).mean():.0%}" for b in [0.15, 0.25, 0.5]))

    lucky = (S[:, 1:] > S[:, :1] + 1e-9).any(1).mean()
    print(f"\nNoise check: {lucky:.1%} of samples have some compressed config that beats the uncompressed one.")
    print(f"\nResults directory: {RESULTS}")


if __name__ == "__main__":
    main()
