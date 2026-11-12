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


def _enumerate(S, C, Q, *, max_stages, final_stage, n_thresholds, cost_mode, candidates, max_first_cost=np.inf):
    """Yield (mean scores, mean costs, make) for every cascade family; make(i) -> Cascade for grid point i."""
    K = S.shape[1]
    cand = list(range(K)) if candidates is None else list(candidates)
    mean_cost = C.mean(0)
    add = (lambda x, y: x + y) if cost_mode == "sum" else np.maximum

    for c in cand:  # single-stage (static) cascades are always in the search space
        yield np.array([S[:, c].mean()]), np.array([mean_cost[c]]), lambda i, c=c: Cascade((c,), ())
    grids = {c: _threshold_grid(Q[:, c], n_thresholds) for c in cand}
    if max_stages >= 2:
        for a, b in product(cand, cand):
            if mean_cost[a] >= mean_cost[b] or mean_cost[a] > max_first_cost:
                continue
            T = grids[a]
            acc = Q[:, a, None] >= T[None, :]                                  # (N, T) accept at stage 1
            sc = np.where(acc, S[:, a, None], S[:, b, None]).mean(0)
            co = np.where(acc, C[:, a, None], add(C[:, a], C[:, b])[:, None]).mean(0)
            yield sc, co, lambda i, a=a, b=b, T=T: Cascade((a, b), (float(T[i]),))
    if max_stages >= 3:
        f = int(np.argmax(mean_cost)) if final_stage is None else final_stage
        small = {c: _threshold_grid(Q[:, c], max(5, n_thresholds // 2)) for c in cand}
        for a, b in product(cand, cand):
            if not (mean_cost[a] < mean_cost[b] < mean_cost[f]) or mean_cost[a] > max_first_cost:
                continue
            T1, T2 = small[a], small[b]
            acc1 = (Q[:, a, None] >= T1[None, :])[:, :, None]                   # (N, T1, 1)
            acc2 = (Q[:, b, None] >= T2[None, :])[:, None, :]                   # (N, 1, T2)
            s3 = np.where(acc1, S[:, a, None, None], np.where(acc2, S[:, b, None, None], S[:, f, None, None]))
            c_ab = add(C[:, a], C[:, b])
            c3 = np.where(acc1, C[:, a, None, None],
                          np.where(acc2, c_ab[:, None, None], add(c_ab, C[:, f])[:, None, None]))
            sc, co = s3.mean(0).ravel(), c3.mean(0).ravel()
            yield sc, co, lambda i, a=a, b=b, T1=T1, T2=T2: Cascade(
                (a, b, f), (float(T1[i // len(T2)]), float(T2[i % len(T2)])))


def fit_cascade(S: np.ndarray, C: np.ndarray, Q: np.ndarray, budget: float, *, max_stages: int = 2,
                final_stage: int | None = None, n_thresholds: int = 21, cost_mode: str = "sum",
                candidates: list[int] | None = None) -> tuple[Cascade, float, float]:
    """Exhaustive search for the best cascade on training data under a mean-cost budget.

    max_stages=2: all (cheap, escalate) pairs and thresholds.
    max_stages=3: (c1, c2, final_stage) with two thresholds; final_stage defaults to the most expensive config.
    Returns (cascade, train score, train cost). Falls back to the cheapest single config if nothing is feasible.
    """
    best: tuple[float, float, Cascade] | None = None
    for scores, costs, make in _enumerate(S, C, Q, max_stages=max_stages, final_stage=final_stage,
                                          n_thresholds=n_thresholds, cost_mode=cost_mode,
                                          candidates=candidates, max_first_cost=budget):
        ok = costs <= budget + 1e-9
        if not ok.any():
            continue
        # best feasible: highest score, then lowest cost
        i = np.lexsort((costs, -np.where(ok, scores, -np.inf)))[0]
        if best is None or (scores[i], -costs[i]) > (best[0], -best[1]):
            best = (float(scores[i]), float(costs[i]), make(i))
    if best is None:
        c = int(np.argmin(C.mean(0)))
        cas = Cascade((c,), ())
        sc, co, _ = simulate(cas, S, C, Q, cost_mode)
        return cas, float(sc.mean()), float(co.mean())
    return best[2], float(best[0]), float(best[1])


def _pareto(scores: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Indices of the (max score, min cost) Pareto frontier, sorted by increasing cost."""
    order = np.lexsort((-scores, costs))
    keep, best = [], -np.inf
    for i in order:
        if scores[i] > best + 1e-12:
            keep.append(i)
            best = scores[i]
    return np.array(keep, int)


def cascade_frontier(S: np.ndarray, C: np.ndarray, Q: np.ndarray, *, max_stages: int = 2,
                     n_thresholds: int = 21, cost_mode: str = "sum",
                     candidates: list[int] | None = None) -> list[tuple[float, float, Cascade]]:
    """Training-set Pareto frontier [(mean score, mean cost, cascade), ...] by increasing cost."""
    sc, co, mk = [], [], []
    for scores, costs, make in _enumerate(S, C, Q, max_stages=max_stages, final_stage=None,
                                          n_thresholds=n_thresholds, cost_mode=cost_mode, candidates=candidates):
        p = _pareto(scores, costs)  # prune per family first: keeps memory small
        sc.append(scores[p])
        co.append(costs[p])
        mk.extend((make, int(i)) for i in p)
    sc, co = np.concatenate(sc), np.concatenate(co)
    return [(float(sc[i]), float(co[i]), mk[i][0](mk[i][1])) for i in _pareto(sc, co)]


def fit_group_cascades(S: np.ndarray, C: np.ndarray, Q: np.ndarray, groups: np.ndarray, budget: float,
                       **kw) -> dict:
    """One cascade per group (e.g. task) under ONE shared average-cost budget.

    Each group contributes its training frontier. Frontiers are merged exactly (request-weighted sums,
    Pareto-pruned after each merge), and the best combination whose overall mean cost fits `budget` wins.
    Returns {group: Cascade}. Groups whose frontier cannot meet the budget jointly fall back to their
    cheapest point.
    """
    names = list(np.unique(groups))
    N = len(S)
    fronts = {g: cascade_frontier(S[groups == g], C[groups == g], Q[groups == g], **kw) for g in names}
    # merged state: arrays of total (score*n, cost*n) and the per-group choice index for each point
    tot_s = np.array([0.0])
    tot_c = np.array([0.0])
    choice = np.zeros((1, 0), int)
    for g in names:
        n = (groups == g).sum()
        fs = np.array([p[0] for p in fronts[g]]) * n
        fc = np.array([p[1] for p in fronts[g]]) * n
        s = (tot_s[:, None] + fs[None]).ravel()
        c = (tot_c[:, None] + fc[None]).ravel()
        ch = np.concatenate([np.repeat(choice, len(fs), 0), np.tile(np.arange(len(fs)), len(tot_s))[:, None]], 1)
        p = _pareto(s, c)
        tot_s, tot_c, choice = s[p], c[p], ch[p]
    ok = tot_c / N <= budget + 1e-9
    k = int(np.argmax(np.where(ok, tot_s, -np.inf))) if ok.any() else 0  # index 0 = cheapest combination
    return {g: fronts[g][choice[k, j]][2] for j, g in enumerate(names)}


def simulate_groups(cascades: dict, S, C, Q, groups, cost_mode: str = "sum"):
    """simulate() with a per-group cascade; returns per-request (score, cost, stage)."""
    score, cost, stage = np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S), int)
    for g, cas in cascades.items():
        m = groups == g
        score[m], cost[m], stage[m] = simulate(cas, S[m], C[m], Q[m], cost_mode)
    return score, cost, stage
