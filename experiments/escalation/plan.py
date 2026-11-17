"""Step 1 (CPU): choose KV-only cascades on the v0.2 requests, to be run end to end on NEW requests.

Only configs with all visual tokens and SnapKV KV compression are allowed, so a retry needs no new prefill
(see uniinfer.runtime.escalation). For each budget we fit, on all 450 v0.2 requests:
  * the best static SnapKV ratio,
  * one two-stage cascade per task (r1 -> r2, threshold on the answer's min log-prob), under the
    memory-freed cost model, with one shared budget.
A 5-fold cross-validated estimate on the same requests is printed for reference.

    python experiments/escalation/plan.py      # writes results/v0.3-escalation/plan.json
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
from analyze import merge_sweeps  # noqa: E402
from cascade_snapkv import cv_group_cascade  # noqa: E402
from common import REPO, load_sweep  # noqa: E402
from run_cascade import confidence_matrices, shuffle_within  # noqa: E402

from uniinfer.controller import best_static_config, fit_group_cascades, simulate_groups  # noqa: E402

RES = REPO / "results"
OUT = RES / "v0.3-escalation"
SIGNAL = "answer min logprob"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="0.075,0.05")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    path = str(RES / "v0.2-baselines" / "oracle_sweep.jsonl")
    extras = [RES / "v0.2-baselines-low" / "oracle_sweep.jsonl", RES / "v0.2-baselines-fine" / "oracle_sweep.jsonl"]
    for k, extra in enumerate(extras):
        merged = OUT / f"merged_sweep{'' if k == len(extras) - 1 else k}.jsonl"
        merge_sweeps(path, extra, merged)
        path = str(merged)
    recs, keys, S, C, ds = load_sweep(path)
    cfgs = recs[0]["configs"]
    cols = [i for i, c in enumerate(cfgs) if c["v_method"] == "full"
            and (c["kv_ratio"] >= 1.0 or c.get("kv_method") == "snapkv")]
    S, C = S[:, cols], C[:, cols]
    keys = [keys[i] for i in cols]
    ratios = [cfgs[i]["kv_ratio"] for i in cols]
    Q = confidence_matrices(recs)[SIGNAL][:, cols]
    print(f"KV-only grid: {len(cols)} configs ({', '.join(keys)})")

    plan = {"signal": SIGNAL, "budgets": {}}
    print("\nCross-validated estimate on the v0.2 requests (5-fold x seeds; raw scores) [realized max-cost]")
    for B in budgets:
        cas = [cv_group_cascade(S, C, Q, ds, 5, sd, [B], "max") for sd in range(a.seeds)]
        shuf = [cv_group_cascade(S, C, shuffle_within(Q, ds, 100 + sd), ds, 5, sd, [B], "max")
                for sd in range(a.seeds)]
        j = best_static_config(S, C, B)
        per = fit_group_cascades(S, C, Q, ds, B, cost_mode="max")
        sc, co, stage = simulate_groups(per, S, C, Q, ds, "max")
        print(f"  B={B}: static {keys[j]} {S[:, j].mean():.3f} [{C[:, j].mean():.3f}] (in-sample) | "
              f"cascade {np.mean([r[B][0].mean() for r in cas]):.3f} [{np.mean([r[B][1].mean() for r in cas]):.3f}]"
              f" | shuffled {np.mean([r[B][0].mean() for r in shuf]):.3f}")
        entry = {"static": {"ratio": ratios[j], "key": keys[j], "fit_score": float(S[:, j].mean()),
                            "fit_cost": float(C[:, j].mean())}, "tasks": {}}
        for d, c in per.items():
            st = [ratios[s] for s in c.stages]
            m = ds == d
            entry["tasks"][d] = {"r1": st[0], "r2": st[1] if len(st) > 1 else None,
                                 "threshold": float(c.thresholds[0]) if c.thresholds else None,
                                 "fit_score": float(sc[m].mean()), "fit_cost": float(co[m].mean()),
                                 "fit_escalation_rate": float((stage[m] > 0).mean())}
            print(f"      {d:8s} {' -> '.join(keys[s] for s in c.stages):45s} thr "
                  f"{entry['tasks'][d]['threshold']}  escalates {(stage[m] > 0).mean():.0%}")
        plan["budgets"][str(B)] = entry
    json.dump(plan, open(OUT / "plan.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'plan.json'}")


if __name__ == "__main__":
    main()
