"""Step 3: static vs per-task vs per-request ORACLE under equal average KV-memory budgets.

Cost of a config = kv_len / kv_len(uncompressed), i.e. relative KV-cache memory.
Runs on CPU in seconds:  python experiments/oracle/analyze.py
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from common import BUDGETS, RESULTS, load_sweep, oracle_score

from uniinfer.controller import best_static_config


def per_task_score(S, C, ds, B):
    tot = 0.0
    for d in np.unique(ds):
        m = ds == d
        tot += S[m, best_static_config(S[m], C[m], B)].sum()
    return tot / len(S)


def bootstrap_score_ci(
    S: np.ndarray,
    C: np.ndarray,
    ds: np.ndarray,
    budgets: Sequence[float],
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Request-bootstrap intervals, shaped (budgets, static/per-task/oracle, low/high).

    Cache the original oracle's lambda-grid decisions: these depend only on a row,
    while feasibility and the winning policy must be recomputed for every resample.
    Static and per-task selection use sampled rows in draw order, preserving the
    original summation order and floating-point tie-breaking. Only oracle policies
    use multiplicity weights. No configurations or tasks are independently resampled.
    If any replicate has no feasible oracle policy, its interval is NaN, as for the
    point estimator; infeasible replicates are never silently discarded.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if S.ndim != 2 or not len(S) or not S.shape[1] or C.shape != S.shape or ds.shape != (len(S),):
        raise ValueError("S and C must be nonempty matching matrices with one task label per row")
    if not np.isfinite(S).all() or not np.isfinite(C).all():
        raise ValueError("scores and costs must be finite")

    # Keep exactly common.oracle_score's grid, tie-breaking and budget tolerance.
    policies = np.array(
        [(S - lam * C - 1e-9 * C).argmax(1) for lam in np.concatenate([[0], np.logspace(-4, 2, 400)])]
    ).T
    policies = np.unique(policies, axis=1)
    rows = np.arange(len(S))[:, None]
    oracle_S, oracle_C = S[rows, policies], C[rows, policies]

    rng = np.random.default_rng(seed)
    samples = rng.integers(len(S), size=(n, len(S)))
    weights = np.array([np.bincount(indices, minlength=len(S)) for indices in samples], dtype=float)
    # Aggregate oracle policies once for all budgets, instead of repeating argmax work.
    means = weights @ np.concatenate([oracle_S, oracle_C], axis=1) / len(S)
    mean_oracle_S, mean_oracle_C = np.split(means, 2, axis=1)
    baselines = np.empty((len(budgets), 2, n))
    for r, indices in enumerate(samples):
        s, c, tasks = S[indices], C[indices], ds[indices]
        for b, budget in enumerate(budgets):
            baselines[b, 0, r] = s[:, best_static_config(s, c, budget)].mean()
            baselines[b, 1, r] = per_task_score(s, c, tasks, budget)

    intervals = np.empty((len(budgets), 3, 2))
    for b, budget in enumerate(budgets):
        static, per_task = baselines[b]
        oracle = np.where(mean_oracle_C <= budget + 1e-9, mean_oracle_S, -np.inf).max(1)
        oracle[oracle == -np.inf] = np.nan
        intervals[b] = np.quantile(
            np.array([static, per_task, oracle]),
            [alpha / 2, 1 - alpha / 2],
            axis=1,
        ).T
    return intervals


def report(S, C, ds, keys, title, intervals=None):
    print(f"\n=== {title} ===")
    widths = (7, 8, 7) if intervals is None else (20, 20, 20)
    columns = "  ".join(f"{name:>{width}s}" for name, width in zip(("static", "per-task", "oracle"), widths))
    print(f"  {'budget':>6s}  {columns}  {'gap':>6s}  static config")
    for b, B in enumerate(BUDGETS):
        j = best_static_config(S, C, B)
        st, pt, orc = S[:, j].mean(), per_task_score(S, C, ds, B), oracle_score(S, C, B)
        if intervals is None:
            print(f"  {B:6.3f}  {st:7.3f}  {pt:8.3f}  {orc:7.3f}  {orc - st:+6.3f}  {keys[j]}")
        else:
            cells = "  ".join(
                f"{score:.3f} [{low:.3f}, {high:.3f}]" for score, (low, high) in zip((st, pt, orc), intervals[b])
            )
            print(f"  {B:6.3f}  {cells}  {orc - st:+6.3f}  {keys[j]}")


def main():
    recs, keys, S, C, ds = load_sweep()
    names = np.unique(ds)
    print(f"samples: {len(S)} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in names) + f"), configs: {len(keys)}")
    print(
        "\nUncompressed score: "
        + ", ".join(f"{d} {S[ds == d, 0].mean():.3f}" for d in names)
        + f", all {S[:, 0].mean():.3f}"
    )

    print("\nMean score per config (sorted by mean cost):")
    print(f"  {'config':30s} {'cost':>5s}  " + "  ".join(f"{d:>7s}" for d in names) + f"  {'all':>6s}")
    for j in np.argsort(C.mean(0)):
        per = "  ".join(f"{S[ds == d, j].mean():7.3f}" for d in names)
        print(f"  {keys[j]:30s} {C[:, j].mean():5.2f}  {per}  {S[:, j].mean():6.3f}")

    report(S, C, ds, keys, "Raw scores")
    Sc = np.minimum(S, S[:, :1])  # no credit for a compressed config beating the uncompressed model
    intervals = bootstrap_score_ci(Sc, C, ds, BUDGETS)
    report(Sc, C, ds, keys, "Luck-free scores (capped at the uncompressed score). Use this for claims.", intervals)

    print("\n=== Sensitivity: cheapest config that keeps the uncompressed score ===")
    for d in names:
        idx = np.where(ds == d)[0]
        cheapest = np.array([C[i, S[i] >= S[i, 0] - 1e-9].min() for i in idx])
        print(
            f"  {d:8s} median cost {np.median(cheapest):.2f}   "
            + "   ".join(f"<= {b:.2f}: {(cheapest <= b + 1e-9).mean():.0%}" for b in [0.15, 0.25, 0.5])
        )

    lucky = (S[:, 1:] > S[:, :1] + 1e-9).any(1).mean()
    print(f"\nNoise check: {lucky:.1%} of samples have some compressed config that beats the uncompressed one.")
    print(f"\nResults directory: {RESULTS}")


if __name__ == "__main__":
    main()
