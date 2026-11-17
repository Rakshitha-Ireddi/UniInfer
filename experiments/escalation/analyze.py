"""Step 3 (CPU): summarize the end-to-end escalation run on new requests.

    python experiments/escalation/analyze.py      # reads results/v0.3-escalation/runs.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uniinfer.stats import bootstrap_ci, paired_diff

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "v0.3-escalation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RES / "runs.jsonl"))
    a = ap.parse_args()
    runs = [json.loads(line) for line in open(a.runs) if line.strip()]
    ds = np.array([r["dataset"] for r in runs])
    names = list(np.unique(ds))
    budgets = sorted({k.split("@")[1] for k in runs[0] if k.startswith("static@")}, key=float, reverse=True)
    print(f"new requests: {len(runs)} (" + ", ".join(f"{d}={int((ds == d).sum())}" for d in names) + ")")

    def col(key, field):
        return np.array([r[key][field] for r in runs], float)

    full = col("full", "score")
    lo, hi = bootstrap_ci(full, groups=ds)
    print(f"uncompressed score {full.mean():.3f} [{lo:.3f}, {hi:.3f}]   "
          + "  ".join(f"{d} {full[ds == d].mean():.3f}" for d in names))
    out = {"n": len(runs), "uncompressed": float(full.mean()), "budgets": {}}

    same = all(r[f"offload@{B}"]["pred"] == r[f"reprefill@{B}"]["pred"] for r in runs for B in budgets)
    print(f"offload and re-prefill give identical answers on every request: {same}")

    for B in budgets:
        st, off, rep = f"static@{B}", f"offload@{B}", f"reprefill@{B}"
        s_st, s_off = col(st, "score"), col(off, "score")
        esc = np.array([r[off]["escalated"] for r in runs])
        d = paired_diff(s_off, s_st, groups=ds)
        res = {
            "static": dict(score=float(s_st.mean()), cost=float(col(st, "cost_max").mean())),
            "cascade": dict(score=float(s_off.mean()), cost_max=float(col(off, "cost_max").mean()),
                            cost_sum=float(col(off, "cost_sum").mean()), escalation_rate=float(esc.mean()),
                            per_task={n: dict(score=float(s_off[ds == n].mean()), static=float(s_st[ds == n].mean()),
                                              escalation_rate=float(esc[ds == n].mean())) for n in names}),
            "cascade_minus_static": d,
        }
        print(f"\n=== budget {float(B) * 100:g}% (plan fit on the v0.2 requests) ===")
        print(f"  static   score {s_st.mean():.3f}  KV cost {res['static']['cost']:.3f}")
        print(f"  cascade  score {s_off.mean():.3f}  KV cost max {res['cascade']['cost_max']:.3f} "
              f"(sum {res['cascade']['cost_sum']:.3f})  escalated {esc.mean():.0%}")
        print(f"  cascade - static: {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]  P(<=0)={d['p_le_0']:.3f}")
        for n in names:
            m = ds == n
            print(f"    {n:8s} static {s_st[m].mean():.3f}  cascade {s_off[m].mean():.3f}  "
                  f"escalated {esc[m].mean():.0%}")

        # latency and memory
        t_st, t_off, t_rep = col(st, "total_s"), col(off, "total_s"), col(rep, "total_s")
        peak_off = col(off, "peak_decode_kv_bytes")
        lat = dict(static_median_s=float(np.median(t_st)), offload_median_s=float(np.median(t_off)),
                   reprefill_median_s=float(np.median(t_rep)))
        if esc.any():
            extra_off = np.array([r[off]["timings"].get("transfer", 0.0) for r in runs])[esc]
            extra_rep = t_rep[esc] - t_off[esc]
            lat.update(escalated_offload_median_s=float(np.median(t_off[esc])),
                       escalated_reprefill_median_s=float(np.median(t_rep[esc])),
                       transfer_median_ms=float(np.median(extra_off) * 1e3),
                       reprefill_extra_median_ms=float(np.median(extra_rep) * 1e3),
                       offload_copy_median_ms=float(np.median([r[off]["timings"].get("offload", 0.0)
                                                               for r in runs]) * 1e3))
        res["latency"] = lat
        res["memory"] = dict(peak_decode_kv_mb_median=float(np.median(peak_off) / 2**20),
                             parked_mb_median=float(np.median(col(off, "offloaded_bytes")) / 2**20))
        print(f"  latency (median s/request): static {lat['static_median_s']:.3f}  cascade+offload "
              f"{lat['offload_median_s']:.3f}  cascade+re-prefill {lat['reprefill_median_s']:.3f}")
        if esc.any():
            print(f"    escalated requests only: offload {lat['escalated_offload_median_s']:.3f} s vs re-prefill "
                  f"{lat['escalated_reprefill_median_s']:.3f} s; PCIe copy back {lat['transfer_median_ms']:.1f} ms vs "
                  f"re-prefill extra {lat['reprefill_extra_median_ms']:.0f} ms; parking copy "
                  f"{lat['offload_copy_median_ms']:.1f} ms per request")
        print(f"  GPU-resident decode KV (median): {res['memory']['peak_decode_kv_mb_median']:.1f} MB; "
              f"parked in host memory: {res['memory']['parked_mb_median']:.1f} MB")
        out["budgets"][B] = res
    json.dump(out, open(Path(a.runs).with_name("escalation_results.json"), "w"), indent=2)
    print(f"\nwrote {Path(a.runs).with_name('escalation_results.json')}")


if __name__ == "__main__":
    main()
