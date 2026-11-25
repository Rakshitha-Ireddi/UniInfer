"""Step 5 (CPU): evaluate the fixed plan on the 1050 held-out requests of the v0.3 scale sweep.

The GPU run (`run.py`) measures cost and latency on 300 new requests, but it only runs static and the
cascade, so its "cascade - static" mixes two effects: task-aware allocation (MMBench drops to 2% KV) and the
per-request confidence signal. The scale sweep (`results/v0.3-scale`) has every KV-only config for 1050
TextVQA/DocVQA/MMBench requests that the plan never saw. Escalation returns exactly the single-config
answers (`run.py --check`), so the plan can be replayed on them exactly and decomposed:

  * static      the plan's static SnapKV ratio
  * task-aware  one SnapKV ratio per task under the same budget, fit on the v0.2 requests like the plan
  * cascade     the plan (per-task r1 -> r2, min log-prob threshold), memory freed before the retry
  * random      the same stages, escalating a random subset of the same size per task (exact expectation:
                same realized cost as the cascade, no per-request information)

It also checks the replay against the GPU run on the requests both contain.

    python experiments/escalation/transfer.py > results/v0.3-escalation/transfer.txt   # also writes transfer.json
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
from common import REPO, load_sweep  # noqa: E402
from run_cascade import confidence_matrices  # noqa: E402

from uniinfer.controller import fit_group_cascades  # noqa: E402
from uniinfer.stats import paired_diff  # noqa: E402

RES = REPO / "results"
OUT = RES / "v0.3-escalation"
SIGNAL = "answer min logprob"


def kv_cols(recs):
    """Column index per KV ratio for the uncompressed + SnapKV configs with all visual tokens."""
    return {c["kv_ratio"]: i for i, c in enumerate(recs[0]["configs"]) if c["v_method"] == "full"
            and c["v_ratio"] >= 1.0 and (c["kv_ratio"] >= 1.0 or c.get("kv_method") == "snapkv")}


def fit_task_aware(B):
    """Per-task single SnapKV ratio under a shared budget, fit on the v0.2 KV-only grid (as plan.py does)."""
    recs, _, S, C, ds = load_sweep(OUT / "merged_sweep.jsonl")
    cols = kv_cols(recs)
    ratios, ix = zip(*sorted(cols.items()))
    S, C = S[:, list(ix)], C[:, list(ix)]
    Q = confidence_matrices(recs)[SIGNAL][:, list(ix)]
    per = fit_group_cascades(S, C, Q, ds, B, cost_mode="max", max_stages=1)
    return {d: ratios[c.stages[0]] for d, c in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(RES / "v0.3-scale" / "oracle_sweep.jsonl"))
    ap.add_argument("--exclude", default=str(RES / "v0.2-baselines" / "oracle_sweep.jsonl"))
    ap.add_argument("--plan", default=str(OUT / "plan.json"))
    ap.add_argument("--runs", default=str(OUT / "runs.jsonl"))
    ap.add_argument("--out", default=str(OUT / "transfer.json"))
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    if not (OUT / "merged_sweep.jsonl").exists():
        raise SystemExit("run experiments/escalation/plan.py first (it writes the merged v0.2 sweep)")
    seen = {json.loads(line)["id"] for line in open(a.exclude)}
    recs, _, S, C, ds = load_sweep(a.sweep)
    tasks = sorted(next(iter(plan["budgets"].values()))["tasks"])
    keep = np.array([r["id"] not in seen and r["dataset"] in tasks for r in recs])
    recs, S, C, ds = [r for r, k in zip(recs, keep) if k], S[keep], C[keep], ds[keep]
    Q = confidence_matrices(recs)[SIGNAL]
    col = kv_cols(recs)
    N = len(S)
    rows = np.arange(N)
    print(f"held-out requests: {N} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in tasks) + "), "
          "none used to fit the plan")
    print(f"uncompressed: {S[:, col[1.0]].mean():.3f}")
    out = {"n": N, "budgets": {}}
    for B, entry in plan["budgets"].items():
        Bf = float(B)
        js = col[entry["static"]["ratio"]]
        st_s, st_c = S[:, js], C[:, js]
        ta_ratio = fit_task_aware(Bf)
        jt = np.array([col[ta_ratio[d]] for d in ds])
        ta_s, ta_c = S[rows, jt], C[rows, jt]
        # cascade replay and the rate-matched random-escalation control
        j1 = np.array([col[entry["tasks"][d]["r1"]] for d in ds])
        r2 = [entry["tasks"][d]["r2"] for d in ds]
        j2 = np.array([col[r] if r is not None else col[entry["tasks"][d]["r1"]] for d, r in zip(ds, r2)])
        thr = np.array([entry["tasks"][d]["threshold"] if entry["tasks"][d]["threshold"] is not None else -np.inf
                        for d in ds])
        esc = Q[rows, j1] < thr
        ca_s = np.where(esc, S[rows, j2], S[rows, j1])
        ca_c = np.where(esc, C[rows, j2], C[rows, j1])  # memory freed before the retry: max(c1, c2) = c2
        p = np.zeros(N)
        for d in tasks:
            m = ds == d
            p[m] = esc[m].mean()
        rnd_s = (1 - p) * S[rows, j1] + p * S[rows, j2]
        rnd_c = (1 - p) * C[rows, j1] + p * C[rows, j2]
        res = {"static": dict(ratio=entry["static"]["ratio"], score=float(st_s.mean()), cost=float(st_c.mean())),
               "task-aware": dict(ratios=ta_ratio, score=float(ta_s.mean()), cost=float(ta_c.mean())),
               "cascade": dict(score=float(ca_s.mean()), cost=float(ca_c.mean()), escalation_rate=float(esc.mean())),
               "random escalation": dict(score=float(rnd_s.mean()), cost=float(rnd_c.mean())),
               "per_task": {}, "paired": {}}
        print(f"\n=== budget {Bf * 100:g}% ===")
        for name in ("static", "task-aware", "cascade", "random escalation"):
            print(f"  {name:18s} score {res[name]['score']:.3f}  KV cost {res[name]['cost']:.3f}")
        print(f"  task-aware ratios: {ta_ratio}; cascade escalates {esc.mean():.0%}")
        for d in tasks:
            m = ds == d
            res["per_task"][d] = dict(static=float(st_s[m].mean()), task_aware=float(ta_s[m].mean()),
                                      cascade=float(ca_s[m].mean()), random=float(rnd_s[m].mean()),
                                      escalation_rate=float(esc[m].mean()))
            print(f"    {d:8s} static {st_s[m].mean():.3f}  task-aware {ta_s[m].mean():.3f}  cascade "
                  f"{ca_s[m].mean():.3f}  random {rnd_s[m].mean():.3f}  escalated {esc[m].mean():.0%}")
        print("  paired differences (95% CI, stratified request bootstrap):")
        for title, x, y in (("cascade - static", ca_s, st_s), ("task-aware - static", ta_s, st_s),
                            ("cascade - task-aware", ca_s, ta_s), ("cascade - random", ca_s, rnd_s)):
            d = paired_diff(x, y, groups=ds)
            res["paired"][title] = d
            print(f"    {title:22s} {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]  P(<=0)={d['p_le_0']:.3f}")
        out["budgets"][B] = res

    # replay vs the GPU run on shared requests
    if Path(a.runs).exists():
        runs = {json.loads(line)["id"]: json.loads(line) for line in open(a.runs)}
        ids = [i for i, r in enumerate(recs) if r["id"] in runs]
        agree = tot = esc_agree = 0
        for B, entry in plan["budgets"].items():
            for i in ids:
                r, run = recs[i], runs[recs[i]["id"]]
                t = entry["tasks"][r["dataset"]]
                c1 = r["configs"][col[t["r1"]]]
                e = t["threshold"] is not None and Q[i, col[t["r1"]]] < t["threshold"]
                pred = r["configs"][col[t["r2"]]]["pred"] if e else c1["pred"]
                stat = r["configs"][col[entry["static"]["ratio"]]]["pred"]
                tot += 2
                agree += (pred == run[f"offload@{B}"]["pred"]) + (stat == run[f"static@{B}"]["pred"])
                esc_agree += e == run[f"offload@{B}"]["escalated"]
        out["replay_check"] = dict(requests=len(ids), answers_identical=int(agree), answers=tot,
                                   escalation_identical=int(esc_agree), escalations=tot // 2)
        print(f"\nreplay vs GPU run on {len(ids)} shared requests x {len(plan['budgets'])} budgets: "
              f"{agree}/{tot} answers identical (static and cascade), "
              f"{esc_agree}/{tot // 2} escalation decisions identical")
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
