"""Step 4: learned per-request selector vs baselines, with request-grouped cross-validation.

Every method is chosen on the TRAINING fold and scored on held-out requests with REAL scores.
Runs on CPU in a few minutes:  python experiments/oracle/predictor.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from common import BUDGETS, RESULTS, config_descriptors, load_sweep, oracle_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from uniinfer.controller import BudgetedSelector, best_static_config


def evaluate(S, C, ds, X, desc, folds, seed, budgets=BUDGETS):
    """Held-out (score, cost) arrays per budget for static, per-task and (if X given) the selector."""
    N = len(S)
    res = {m: {B: [np.zeros(N), np.zeros(N)] for B in budgets} for m in ["static", "per-task", "selector"]}

    def put(method, B, rows, j):
        res[method][B][0][rows] = S[rows, j]
        res[method][B][1][rows] = C[rows, j]

    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(S):
        sel = BudgetedSelector(desc, seed=seed).fit(X[tr], S[tr], C[tr]) if X is not None else None
        for B in budgets:
            put("static", B, te, best_static_config(S[tr], C[tr], B))
            for d in np.unique(ds):
                trd, ted = tr[ds[tr] == d], te[ds[te] == d]
                put("per-task", B, ted, best_static_config(S[trd], C[trd], B))
            if sel is not None:
                put("selector", B, te, sel.select(X[te], C[te], B))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    recs, keys, S, C, ds = load_sweep()
    F = np.load(RESULTS / "features.npz")
    pos = {i: k for k, i in enumerate(F["ids"])}
    order = np.array([pos[r["id"]] for r in recs])
    t1 = F["t1"][order]
    t2 = F["t2"][order].astype(np.float32)
    t3 = F["t3"][order].astype(np.float32)
    desc = config_descriptors(recs)

    def pca(Z, k):  # unsupervised, no labels
        return PCA(k, random_state=0).fit_transform(Z)

    feature_sets = {
        "task label only": (ds[:, None] == np.unique(ds)[None]).astype(np.float32),
        "T1 cheap stats": t1,
        "T2 +question emb": np.concatenate([t1, pca(t2, 16)], 1),
        "T3 +prefill hidden": np.concatenate([t1, pca(t2, 16), pca(t3, 16)], 1),
    }
    med = np.median(F["times"][order], 0)
    print(f"N={len(S)} configs={S.shape[1]} folds={a.folds} seeds={a.seeds}; uncompressed score {S[:, 0].mean():.3f}")
    print(f"median per-request overhead: vision encoder {med[0] * 1e3:.0f} ms (paid anyway), "
          f"T1/T2 features {med[1] * 1e3:.1f} ms, T3 full prefill {med[2] * 1e3:.0f} ms")

    table = {}
    base = [evaluate(S, C, ds, None, desc, a.folds, sd) for sd in range(a.seeds)]
    for m in ["static", "per-task"]:
        table[m] = {B: (np.mean([r[m][B][0].mean() for r in base]), np.std([r[m][B][0].mean() for r in base]),
                        np.mean([r[m][B][1].mean() for r in base])) for B in BUDGETS}
    for name, X in feature_sets.items():
        runs = [evaluate(S, C, ds, X, desc, a.folds, sd) for sd in range(a.seeds)]
        table[name] = {B: (np.mean([r["selector"][B][0].mean() for r in runs]),
                           np.std([r["selector"][B][0].mean() for r in runs]),
                           np.mean([r["selector"][B][1].mean() for r in runs])) for B in BUDGETS}
    table["oracle (upper bound)"] = {B: (oracle_score(S, C, B), 0.0, B) for B in BUDGETS}

    cols = list(table)
    print("\nHeld-out score (mean ± std over seeds) [realized average cost]")
    print(f"{'budget':>6s}  " + "  ".join(f"{c:>22s}" for c in cols))
    for B in BUDGETS:
        print(f"{B:6.3f}  " + "  ".join(f"{table[c][B][0]:.3f}±{table[c][B][1]:.3f} [{table[c][B][2]:.2f}]".rjust(22)
                                       for c in cols))
    json.dump({c: {str(B): dict(score=float(v[0]), std=float(v[1]), cost=float(v[2])) for B, v in table[c].items()}
               for c in cols}, open(RESULTS / "predictor_results.json", "w"), indent=2)
    print(f"\nwrote {RESULTS / 'predictor_results.json'}")


if __name__ == "__main__":
    main()
