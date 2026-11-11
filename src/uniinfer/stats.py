"""Bootstrap statistics over requests.

Every replicate is a vector of resampling COUNTS over requests (rows). A replicate's mean of any
per-request quantity is then `W @ x / N`, so thousands of replicates cost one matrix multiply. This makes
it cheap to re-run whole selection procedures (best static config, oracle) inside each replicate.

Resampling is stratified by `groups` (e.g. dataset) when given, so every replicate keeps the design's
per-dataset sample sizes.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_weights(n: int, n_boot: int = 2000, seed: int = 0,
                      groups: Sequence | np.ndarray | None = None) -> np.ndarray:
    """(n_boot, n) resampling counts. Each row sums to n (and to each group's size within that group)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    groups = np.asarray(groups)
    W = np.zeros((n_boot, n))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
    return W


def percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci(values: Sequence[float] | np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 groups: Sequence | np.ndarray | None = None) -> tuple[float, float]:
    """Percentile CI of the mean of per-request `values`."""
    x = np.asarray(values, float)
    W = bootstrap_weights(len(x), n_boot, seed, groups)
    return percentile_ci(W @ x / len(x), alpha)


def paired_diff(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, n_boot: int = 2000,
                alpha: float = 0.05, seed: int = 0, groups: Sequence | np.ndarray | None = None) -> dict:
    """Paired bootstrap of mean(a) - mean(b) over the SAME requests.

    Returns diff, its CI, and p_le_0, the bootstrap probability that the difference is <= 0
    (a one-sided measure of how often a fails to beat b).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    W = bootstrap_weights(len(d), n_boot, seed, groups)
    reps = W @ d / len(d)
    lo, hi = percentile_ci(reps, alpha)
    return {"diff": float(d.mean()), "low": lo, "high": hi, "p_le_0": float((reps <= 0).mean())}


def best_static_replicates(S: np.ndarray, C: np.ndarray, budget: float, W: np.ndarray) -> np.ndarray:
    """Score of the best budget-feasible single config, RE-SELECTED inside every replicate."""
    n = S.shape[0]
    mean_s, mean_c = W @ S / n, W @ C / n                          # (n_boot, K)
    feasible = mean_c <= budget + 1e-9
    masked = np.where(feasible, mean_s, -np.inf)
    out = masked.max(1)
    none = ~feasible.any(1)                                       # fallback: cheapest config
    if none.any():
        out[none] = mean_s[none, mean_c[none].argmin(1)]
    return out


def oracle_replicates(S: np.ndarray, C: np.ndarray, budget: float, W: np.ndarray,
                      lambdas: np.ndarray | None = None) -> np.ndarray:
    """Per-request oracle under an average budget, re-solved inside every replicate (Lagrangian sweep).

    For each lambda, every request's choice is fixed (argmax S - lambda*C), so per-lambda per-request
    score/cost vectors are precomputed once; each replicate then picks the best feasible lambda.
    """
    lambdas = np.concatenate([[0.0], np.logspace(-4, 2, 400)]) if lambdas is None else lambdas
    n = S.shape[0]
    rows = np.arange(n)
    choice = np.stack([(S - lam * C - 1e-9 * C).argmax(1) for lam in lambdas])   # (L, n)
    s_l, c_l = S[rows, choice], C[rows, choice]                                   # (L, n)
    rep_s, rep_c = W @ s_l.T / n, W @ c_l.T / n                                   # (n_boot, L)
    return np.where(rep_c <= budget + 1e-9, rep_s, -np.inf).max(1)
