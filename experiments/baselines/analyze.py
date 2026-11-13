"""v0.2 baselines: does the oracle headroom survive stronger compression operators?

Inputs: results/v0.2-baselines/oracle_sweep.jsonl from
    python experiments/oracle/sweep.py --n 150 --v-methods uniform,dissim,attn --kv-methods l2norm,snapkv \
        --out results/v0.2-baselines/oracle_sweep.jsonl
Runs on CPU in about a minute:  python experiments/baselines/analyze.py

Questions
  1. How strong is each operator on its own (visual-only and KV-only), per task?
  2. With SnapKV / FastV-style pruning available to the STATIC baseline, how much oracle headroom is left?
  3. Does task-aware allocation still beat the best static config on the richer grid (held-out)?
All scores are luck-free (capped at the uncompressed score) except where marked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from common import BUDGETS, REPO, config_descriptors, config_key, load_sweep, oracle_score  # noqa: E402
from predictor import evaluate  # noqa: E402

from uniinfer.controller import best_static_config  # noqa: E402
from uniinfer.stats import (  # noqa: E402
    best_static_replicates,
    bootstrap_weights,
    oracle_replicates,
    paired_diff,
    percentile_ci,
)

OUT = REPO / "results" / "v0.2-baselines"
LOW_BUDGETS = [0.10, 0.075, 0.05]

# Operator families available to every method (static, task-aware, oracle). "full" is always included.
FAMILIES = {
    "v0.1 operators (uniform/dissim + L2-norm)": lambda c: c["v_method"] != "attn" and _km(c) == "l2norm",
    "+ SnapKV": lambda c: c["v_method"] != "attn",
    "+ FastV-style visual": lambda c: _km(c) == "l2norm",
    "all operators": lambda c: True,
}


def _km(c):
    return "l2norm" if c["kv_ratio"] >= 1.0 else c.get("kv_method", "l2norm")


def merge_sweeps(main, extra, out):
    """Append the configs of a second sweep over the same requests (e.g. lower SnapKV ratios) to the first.
    Configs already present (same key, e.g. every 'kv@1.00') are kept from the main sweep."""
    ext = {r["id"]: r for r in (json.loads(line) for line in open(extra) if line.strip())}
    n = 0
    with open(out, "w") as f:
        for line in open(main):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in ext:
                continue
            have = {config_key(c) for c in r["configs"]}
            r["configs"] += [c for c in ext[r["id"]]["configs"] if config_key(c) not in have]
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"merged {n} requests from {main} + {extra}")


def operator_table(recs, S, ds):
    """Mean luck-free score of single-axis compression: visual-only (KV 100%) and KV-only (visual 100%)."""
    cfgs = recs[0]["configs"]
    names = list(np.unique(ds))
    rows = {}
    ratios = sorted(({c["v_ratio"] for c in cfgs} | {c["kv_ratio"] for c in cfgs}) - {1.0}, reverse=True)
    for vm in ["uniform", "dissim", "attn"]:
        for r in ratios:
            j = [i for i, c in enumerate(cfgs) if c["v_method"] == vm and c["v_ratio"] == r and c["kv_ratio"] == 1.0]
            if j:
                rows[f"visual {vm:7s} keep {r:.2f}"] = j[0]
    for km in ["l2norm", "snapkv"]:
        for r in ratios:
            j = [i for i, c in enumerate(cfgs) if c["v_method"] == "full" and _km(c) == km and c["kv_ratio"] == r]
            if j:
                rows[f"KV     {km:7s} keep {r:.2f}"] = j[0]
    table = {k: {"all": float(S[:, j].mean()), **{d: float(S[ds == d, j].mean()) for d in names}}
             for k, j in rows.items()}
    print("\n=== 1. Operators on their own (luck-free mean score; uncompressed in the first row) ===")
    print(f"  {'operator':28s} {'all':>6s} " + " ".join(f"{d:>8s}" for d in names))
    print(f"  {'uncompressed':28s} {S[:, 0].mean():6.3f} " + " ".join(f"{S[ds == d, 0].mean():8.3f}" for d in names))
    for k, row in table.items():
        print(f"  {k:28s} {row['all']:6.3f} " + " ".join(f"{row[d]:8.3f}" for d in names))
    return table


def headroom(recs, S, C, ds, keys, n_boot, seed, budgets):
    """Best static vs oracle per operator family, with stratified bootstrap CIs (selection redone per replicate)."""
    cfgs = recs[0]["configs"]
    W = bootstrap_weights(len(S), n_boot, seed, groups=ds)
    out = {}
    print(f"\n=== 2. Oracle headroom per operator family (luck-free; 95% CI, {n_boot} stratified replicates) ===")
    for fam, keep in FAMILIES.items():
        cols = np.array([i for i, c in enumerate(cfgs) if keep(c)])
        Sf, Cf = S[:, cols], C[:, cols]
        print(f"  {fam}  ({len(cols)} configs)")
        out[fam] = {}
        for B in budgets:
            if Cf.mean(0).min() > B + 1e-9:   # no single config this cheap: the family cannot reach B
                out[fam][str(B)] = dict(infeasible=True)
                print(f"    B={B:5.3f}  infeasible (cheapest config costs {Cf.mean(0).min():.3f})")
                continue
            j = best_static_config(Sf, Cf, B)
            st, orc = float(Sf[:, j].mean()), float(oracle_score(Sf, Cf, B))
            st_r, or_r = best_static_replicates(Sf, Cf, B, W), oracle_replicates(Sf, Cf, B, W)
            (sl, sh), (gl, gh) = percentile_ci(st_r), percentile_ci(or_r - st_r)
            out[fam][str(B)] = dict(static=st, static_ci=[sl, sh], oracle=orc, gap=orc - st, gap_ci=[gl, gh],
                                    static_config=keys[cols[j]])
            print(f"    B={B:5.3f}  static {st:.3f} [{sl:.3f}, {sh:.3f}]  oracle {orc:.3f}  "
                  f"gap {orc - st:+.3f} [{gl:+.3f}, {gh:+.3f}]   static = {keys[cols[j]]}")
    return out


def task_aware(recs, S, C, ds, folds, seeds, budgets):
    """Held-out static / per-task / task-aware selector for the v0.1 family vs all operators (raw scores, as v0.1).
    Task-aware minus static is a paired, stratified bootstrap over requests (per-request scores averaged over seeds).
    Budgets below the family's cheapest config are skipped."""
    cfgs = recs[0]["configs"]
    onehot = (ds[:, None] == np.unique(ds)[None]).astype(np.float32)
    out = {}
    print(f"\n=== 3. Held-out allocation (raw scores, {folds}-fold x {seeds} seeds)  score [realized cost] ===")
    for fam in ["v0.1 operators (uniform/dissim + L2-norm)", "all operators"]:
        cols = np.array([i for i, c in enumerate(cfgs) if FAMILIES[fam](c)])
        sub = [dict(r, configs=[r["configs"][i] for i in cols]) for r in recs[:1]]
        desc = config_descriptors(sub)
        fb = [B for B in budgets if C[:, cols].mean(0).min() <= B + 1e-9]
        runs = [evaluate(S[:, cols], C[:, cols], ds, onehot, desc, folds, sd, budgets=fb) for sd in range(seeds)]
        out[fam] = {}
        for m, label in [("static", "static"), ("per-task", "per-task static"), ("selector", "task-aware")]:
            row = {str(B): dict(score=float(np.mean([r[m][B][0].mean() for r in runs])),
                                cost=float(np.mean([r[m][B][1].mean() for r in runs]))) for B in fb}
            out[fam][label] = row
            cells = "".join(f"  B={B}: {row[str(B)]['score']:.3f} [{row[str(B)]['cost']:.2f}]" for B in fb)
            print(f"  {fam[:24]:24s} {label:16s}{cells}")
        diffs = {}
        for B in fb:
            per_req = {m: np.mean([r[m][B][0] for r in runs], 0) for m in ("selector", "static")}
            d = paired_diff(per_req["selector"], per_req["static"], groups=ds)
            diffs[str(B)] = d
            print(f"    task-aware - static  B={B}: {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
                  f"  P(<=0)={d['p_le_0']:.3f}")
        out[fam]["task-aware minus static"] = diffs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(OUT / "oracle_sweep.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--extra", help="second sweep over the same requests to merge in (e.g. low SnapKV ratios)")
    ap.add_argument("--out", help="results JSON (default: baselines_results.json next to the sweep)")
    a = ap.parse_args()
    budgets = list(BUDGETS)
    if a.extra:
        merged = Path(a.extra).with_name("merged_sweep.jsonl")
        merge_sweeps(a.sweep, a.extra, merged)
        a.sweep, budgets = str(merged), budgets + LOW_BUDGETS
    recs, keys, S_raw, C, ds = load_sweep(a.sweep)
    S = np.minimum(S_raw, S_raw[:, :1])
    names = np.unique(ds)
    print(f"samples: {len(S)} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in names) + f"), configs: {len(keys)}")
    res = dict(uncompressed={"all": float(S[:, 0].mean()), **{d: float(S[ds == d, 0].mean()) for d in names}},
               operators=operator_table(recs, S, ds),
               headroom=headroom(recs, S, C, ds, keys, a.n_boot, a.seed, budgets),
               allocation=task_aware(recs, S_raw, C, ds, a.folds, a.seeds, budgets),
               n=len(S), n_configs=len(keys))
    out = Path(a.out) if a.out else Path(a.sweep).with_name("baselines_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
