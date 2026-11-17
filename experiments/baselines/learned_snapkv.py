"""Per-request learned selection on the SnapKV grid, and a "needs almost no KV" detector (CPU only).

v0.1 found that learned selectors (T1-T3 request features) did no better than a task label, but that was
on a grid with weak operators. Here the grid has SnapKV down to 2% KV (149 configs), and we ask:
  1. Near SnapKV's cliff, does a learned per-request selector beat task-aware allocation (held-out)?
  2. Can request features tell, BEFORE answering, which requests keep their answer at about 2-3% KV?
     Measured within each task, so that predicting the task label alone earns nothing.

    python experiments/baselines/learned_snapkv.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
sys.path.insert(0, str(HERE))  # this directory first: oracle/ also has an analyze.py
from analyze import merge_sweeps  # noqa: E402
from common import REPO, config_descriptors, load_sweep  # noqa: E402
from predictor import evaluate  # noqa: E402

from uniinfer.stats import paired_diff  # noqa: E402

RES = REPO / "results"
BASE = RES / "v0.2-baselines" / "oracle_sweep.jsonl"
EXTRA = [RES / "v0.2-baselines-low" / "oracle_sweep.jsonl", RES / "v0.2-baselines-fine" / "oracle_sweep.jsonl"]
FEATURES = RES / "v0.1-oracle" / "features.npz"
OUT = RES / "v0.2-baselines-fine" / "learned_snapkv"
BUDGETS = [0.15, 0.10, 0.075, 0.05, 0.04, 0.03]
CHEAP = ["full@1.00|snapkv@0.02", "full@1.00|snapkv@0.03"]   # "almost no KV"


def feature_sets(recs, ds):
    F = np.load(FEATURES)
    pos = {i: k for k, i in enumerate(F["ids"])}
    order = np.array([pos[r["id"]] for r in recs])
    t1 = F["t1"][order]
    t2 = F["t2"][order].astype(np.float32)
    t3 = F["t3"][order].astype(np.float32)

    def pca(Z, k):  # unsupervised, no labels
        return PCA(k, random_state=0).fit_transform(Z)

    onehot = (ds[:, None] == np.unique(ds)[None]).astype(np.float32)
    return {
        "task label only": onehot,
        "T1 cheap stats": np.concatenate([onehot, t1], 1),
        "T2 +question emb": np.concatenate([onehot, t1, pca(t2, 16)], 1),
        "T3 +prefill hidden": np.concatenate([onehot, t1, pca(t2, 16), pca(t3, 16)], 1),
    }, np.median(F["times"][order], 0)


def selection(S, C, ds, feats, desc, folds, seeds, budgets):
    print(f"\n=== 1. Held-out selection on the SnapKV grid (raw scores, {folds}-fold x {seeds} seeds) ===")
    runs = {name: [evaluate(S, C, ds, X, desc, folds, sd, budgets=budgets) for sd in range(seeds)]
            for name, X in feats.items()}
    rows = {"static": [{B: r["static"][B] for B in budgets} for r in runs["task label only"]],
            "per-task static": [{B: r["per-task"][B] for B in budgets} for r in runs["task label only"]]}
    rows.update({name: [{B: r["selector"][B] for B in budgets} for r in rr] for name, rr in runs.items()})

    def held(rr, B):
        return np.mean([r[B][0] for r in rr], 0), np.mean([r[B][1] for r in rr], 0)

    table = {}
    print(f"  {'method':20s}" + "".join(f"{'B=' + str(B):>16s}" for B in budgets))
    for name, rr in rows.items():
        table[name] = {str(B): dict(score=float(held(rr, B)[0].mean()), cost=float(held(rr, B)[1].mean()))
                       for B in budgets}
        print(f"  {name:20s}" + "".join(f"{held(rr, B)[0].mean():.3f} [{held(rr, B)[1].mean():.3f}]".rjust(16)
                                        for B in budgets))
    paired = {}
    print("\n  Paired difference vs the task-label selector (95% CI, stratified bootstrap):")
    for name in ["T1 cheap stats", "T2 +question emb", "T3 +prefill hidden"]:
        paired[name] = {}
        cells = []
        for B in budgets:
            d = paired_diff(held(rows[name], B)[0], held(rows["task label only"], B)[0], groups=ds)
            d["cost_diff"] = float((held(rows[name], B)[1] - held(rows["task label only"], B)[1]).mean())
            paired[name][str(B)] = d
            cells.append(f"{d['diff']:+.3f} [{d['low']:+.3f},{d['high']:+.3f}]")
        print(f"  {name:20s} " + "  ".join(f"B={B}: {c}" for B, c in zip(budgets, cells)))
    return table, paired


def detector(recs, S, ds, feats, keys, folds, seeds):
    """Within-task AUROC of predicting 'this request keeps its full answer at a ~2-3% KV config'.

    Two populations: all requests (a request the model already gets wrong trivially 'keeps' its score, and
    needs no KV either), and only requests the uncompressed model answers with a nonzero score."""
    Sc = np.minimum(S, S[:, :1])
    print("\n=== 2. 'Needs almost no KV' detector: within-task AUROC (5-fold CV x seeds; 0.5 = chance) ===")
    out = {}
    for cheap in CHEAP:
        j = keys.index(cheap)
        y = (Sc[:, j] >= Sc[:, 0] - 1e-9).astype(int)
        ok = S[:, 0] > 0
        print(f"  label: keeps the answer at {cheap}   base rate (all | correct-only) "
              + ", ".join(f"{d} {y[ds == d].mean():.0%} | {y[(ds == d) & ok].mean():.0%}" for d in np.unique(ds)))
        out[cheap] = {"base_rate": {d: float(y[ds == d].mean()) for d in np.unique(ds)}}
        for name, X in feats.items():
            if name == "task label only":
                continue
            p = np.zeros(len(y))
            aucs = {d: [] for d in np.unique(ds)}
            aucs_ok = {d: [] for d in np.unique(ds)}
            for sd in range(seeds):
                strat = np.char.add(ds.astype(str), y.astype(str))     # stratify by task and label
                for tr, te in StratifiedKFold(folds, shuffle=True, random_state=sd).split(X, strat):
                    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                         l2_regularization=1.0, random_state=sd)
                    p[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                for d in np.unique(ds):
                    for acc, m in ((aucs, ds == d), (aucs_ok, (ds == d) & (S[:, 0] > 0))):
                        if 0 < y[m].mean() < 1:
                            acc[d].append(roc_auc_score(y[m], p[m]))
            row = {d: (float(np.mean(v)) if v else float("nan")) for d, v in aucs.items()}
            row_ok = {d: (float(np.mean(v)) if v else float("nan")) for d, v in aucs_ok.items()}
            out[cheap][name] = dict(all=row, correct_only=row_ok)
            print(f"    {name:20s} all: " + "  ".join(f"{d} {row[d]:.3f}" for d in row)
                  + "   | correct-only: " + "  ".join(f"{d} {row_ok[d]:.3f}" for d in row_ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    path = str(BASE)
    for k, extra in enumerate(EXTRA):
        merged = OUT.with_name(f"merged_sweep{'' if k == len(EXTRA) - 1 else k}.jsonl")
        merge_sweeps(path, extra, merged)
        path = str(merged)
    recs, keys, S, C, ds = load_sweep(path)
    feats, med = feature_sets(recs, ds)
    print(f"N={len(S)} configs={S.shape[1]}; median overhead: T1/T2 features {med[1] * 1e3:.1f} ms, "
          f"T3 (full prefill) {med[2] * 1e3:.0f} ms")
    table, paired = selection(S, C, ds, feats, config_descriptors(recs), a.folds, a.seeds, BUDGETS)
    det = detector(recs, S, ds, feats, keys, a.folds, a.seeds)
    json.dump(dict(budgets=BUDGETS, table=table, paired_vs_task_label=paired, detector=det),
              open(str(OUT) + ".json", "w"), indent=2)
    print(f"\nwrote {OUT}.json")


if __name__ == "__main__":
    main()
