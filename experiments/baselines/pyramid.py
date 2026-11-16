"""PyramidKV vs SnapKV at the same average KV memory (paired, same requests, same visual config).

PyramidKV keeps more tokens in lower layers and fewer in upper layers (beta = 20); SnapKV keeps the same
number in every layer. Both use the same per-head selection, so the comparison isolates the per-layer split.

    python experiments/baselines/pyramid.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from common import REPO, config_key  # noqa: E402

from uniinfer.stats import paired_diff  # noqa: E402

RES = REPO / "results"
SNAP = [RES / "v0.2-baselines" / "oracle_sweep.jsonl", RES / "v0.2-baselines-low" / "oracle_sweep.jsonl"]
PYR = RES / "v0.2-pyramid" / "oracle_sweep.jsonl"


def load(paths):
    out = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                d = out.setdefault(r["id"], {"dataset": r["dataset"], "cfg": {}})
                for c in r["configs"]:
                    d["cfg"][config_key(c)] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyramid", default=str(PYR))
    ap.add_argument("--out", default=str(PYR.with_name("pyramid_results.json")))
    a = ap.parse_args()
    snap, pyr = load(SNAP), load([a.pyramid])
    ids = [i for i in pyr if i in snap]
    ds = np.array([pyr[i]["dataset"] for i in ids])
    names = list(np.unique(ds))
    print(f"requests: {len(ids)} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in names) + ")")
    vis = sorted({k.split("|")[0] for k in pyr[ids[0]]["cfg"]}, key=lambda v: (v != "full@1.00", v))
    ratios = sorted({k.split("@")[-1] for k in pyr[ids[0]]["cfg"] if "pyramid@" in k}, reverse=True)
    res = {}
    print("\nluck-free score (capped at the uncompressed score); difference = PyramidKV - SnapKV, paired 95% CI")
    print(f"  {'config':22s} {'SnapKV':>7s} {'Pyramid':>8s} {'diff [95% CI]':>26s} {'cost ratio':>11s}   per dataset")
    for v in vis:
        for r in ratios:
            ks, kp = f"{v}|snapkv@{r}", f"{v}|pyramid@{r}"
            full = np.array([snap[i]["cfg"]["full@1.00|kv@1.00"]["score"] for i in ids])
            s = np.minimum([snap[i]["cfg"][ks]["score"] for i in ids], full)
            p = np.minimum([pyr[i]["cfg"][kp]["score"] for i in ids], full)
            cost = np.mean([pyr[i]["cfg"][kp]["kv_len"] / snap[i]["cfg"][ks]["kv_len"] for i in ids])
            d = paired_diff(p, s, groups=ds)
            per = {n: float((p - s)[ds == n].mean()) for n in names}
            res[f"{v}|{r}"] = dict(snapkv=float(s.mean()), pyramid=float(p.mean()), cost_ratio=float(cost), **d,
                                   per_dataset=per)
            print(f"  {v + ' KV ' + r:22s} {s.mean():7.3f} {p.mean():8.3f} "
                  f"{d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]".ljust(68) + f"{cost:6.3f}   "
                  + "  ".join(f"{n} {per[n]:+.3f}" for n in names))
    json.dump(dict(n=len(ids), results=res), open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
