"""Confidence cascades: answer with a cheap config, escalate when the model is unsure (ADR-0008).

Notation (as in selector.py): S[i,c] quality, C[i,c] relative KV cost, Q[i,c] a confidence signal
(higher = more confident) observed after running config c on request i.

A cascade is a ladder of configs (c_1, ..., c_m) with thresholds (t_1, ..., t_{m-1}). Request i runs
c_1. If Q[i,c_1] >= t_1 it stops; otherwise it runs c_2, and so on. The last stage always answers.

Cost of a request:
  "sum" (default, conservative): sum of the costs of every stage attempted. The failed attempts count.
  "max": cost of the most expensive stage attempted (memory is freed before retrying).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Cascade:
    stages: tuple[int, ...]         # config indices, cheapest first
    thresholds: tuple[float, ...]   # len(stages) - 1

    def __post_init__(self):
        assert len(self.thresholds) == len(self.stages) - 1


def simulate(cascade: Cascade, S: np.ndarray, C: np.ndarray, Q: np.ndarray,
             cost_mode: str = "sum") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-request (score, cost, stage index answered at)."""
    N = len(S)
    rows = np.arange(N)
    active = np.ones(N, bool)
    score, cost = np.zeros(N), np.zeros(N)
    stage = np.full(N, len(cascade.stages) - 1)
    for k, c in enumerate(cascade.stages):
        cost[active] = cost[active] + C[active, c] if cost_mode == "sum" else np.maximum(cost[active], C[active, c])
        last = k == len(cascade.stages) - 1
        accept = active & (np.ones(N, bool) if last else Q[:, c] >= cascade.thresholds[k])
        score[accept] = S[rows[accept], c]
        stage[accept] = k
        active &= ~accept
    return score, cost, stage


def _threshold_grid(q: np.ndarray, n: int) -> np.ndarray:
    qs = np.unique(np.quantile(q[np.isfinite(q)], np.linspace(0, 1, n)))
    return np.r_[-np.inf, qs, np.inf]  # -inf: never escalate; +inf: always escalate


def fit_cascade(S: np.ndarray, C: np.ndarray, Q: np.ndarray, budget: float, *, max_stages: int = 2,
                final_stage: int | None = None, n_thresholds: int = 21, cost_mode: str = "sum",
                candidates: list[int] | None = None) -> tuple[Cascade, float, float]:
    """Exhaustive search for the best cascade on training data under a mean-cost budget.

    max_stages=2: all (cheap, escalate) pairs and thresholds.
    max_stages=3: (c1, c2, final_stage) with two thresholds; final_stage defaults to the most expensive config.
    Returns (cascade, train score, train cost). Falls back to the cheapest single config if nothing is feasible.
    """
    K = S.shape[1]
    cand = list(range(K)) if candidates is None else list(candidates)
    mean_cost = C.mean(0)
    best: tuple[float, float, Cascade] | None = None

    def consider(cas: Cascade):
        nonlocal best
        sc, co, _ = simulate(cas, S, C, Q, cost_mode)
        if co.mean() <= budget + 1e-9 and (best is None or (sc.mean(), -co.mean()) > (best[0], -best[1])):
            best = (sc.mean(), co.mean(), cas)

    for c in cand:  # single-stage (static) cascades are always in the search space
        consider(Cascade((c,), ()))
    grids = {c: _threshold_grid(Q[:, c], n_thresholds) for c in cand}
    if max_stages >= 2:
        for a, b in product(cand, cand):
            if mean_cost[a] >= mean_cost[b] or mean_cost[a] > budget:
                continue
            for t in grids[a]:
                consider(Cascade((a, b), (float(t),)))
    if max_stages >= 3:
        f = int(np.argmax(mean_cost)) if final_stage is None else final_stage
        small = {c: _threshold_grid(Q[:, c], max(5, n_thresholds // 2)) for c in cand}
        for a, b in product(cand, cand):
            if not (mean_cost[a] < mean_cost[b] < mean_cost[f]) or mean_cost[a] > budget:
                continue
            for t1, t2 in product(small[a], small[b]):
                consider(Cascade((a, b, f), (float(t1), float(t2))))
    if best is None:
        c = int(np.argmin(mean_cost))
        cas = Cascade((c,), ())
        sc, co, _ = simulate(cas, S, C, Q, cost_mode)
        return cas, float(sc.mean()), float(co.mean())
    return best[2], float(best[0]), float(best[1])
