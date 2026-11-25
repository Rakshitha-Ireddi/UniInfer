"""Figure for the v0.3 escalation study (light and dark).

Left: the fixed plan replayed on 1050 held-out requests (`transfer.json`). Right: measured latency on the GPU
run (`escalation_results.json`), for requests that escalate.

    python experiments/escalation/figures.py      # docs/figures/escalation_{light,dark}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from common import REPO  # noqa: E402
from make_figures import THEMES, style  # noqa: E402

RES = REPO / "results" / "v0.3-escalation"
OUT = REPO / "docs" / "figures"


def _legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def fig(theme, t, tr, lat, out):
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, facecolor=t["surface"],
                               gridspec_kw=dict(width_ratios=[1.25, 1]))
    budgets = list(tr["budgets"])
    methods = [("static", "Static SnapKV", t["muted"]), ("task-aware", "Task-aware", t["series"][3]),
               ("random escalation", "Cascade stages, random escalation", t["series"][1]),
               ("cascade", "Confidence cascade", t["series"][0])]
    style(a1, t)
    w = 0.2
    for k, (m, label, col) in enumerate(methods):
        y = [tr["budgets"][b][m]["score"] for b in budgets]
        x = np.arange(len(budgets)) + (k - 1.5) * w
        a1.bar(x, y, w * 0.92, color=col, label=label, zorder=3)
        for xi, yi in zip(x, y):
            a1.text(xi, yi + 0.004, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
    a1.set_xticks(range(len(budgets)))
    a1.set_xticklabels([f"{float(b) * 100:g}% KV budget" for b in budgets])
    lo = min(tr["budgets"][b]["static"]["score"] for b in budgets)
    a1.set_ylim(lo - 0.06, max(tr["budgets"][b]["cascade"]["score"] for b in budgets) + 0.04)
    a1.set_ylabel("Held-out score (axis does not start at 0)", color=t["ink2"], fontsize=10)
    a1.set_title(f"Fixed plan on {tr['n']} held-out requests", color=t["ink"], fontsize=10.5, loc="left")
    _legend(a1, t, loc="upper right", ncol=1)

    style(a2, t)
    names = [("static_mean_s", "Static"), ("escalated_offload_median_s", "Escalated:\nparked cache"),
             ("escalated_reprefill_median_s", "Escalated:\nre-prefill")]
    cols = [t["muted"], t["series"][0], t["series"][1]]
    for k, b in enumerate(budgets):
        L = lat["budgets"][b]["latency"]
        y = [L["static_median_s"], L["escalated_offload_median_s"], L["escalated_reprefill_median_s"]]
        x = np.arange(3) + (k - 0.5) * 0.36
        a2.bar(x, y, 0.33, color=cols, alpha=1.0 if k == 0 else 0.6, zorder=3,
               label=f"{float(b) * 100:g}% budget")
        for xi, yi in zip(x, y):
            a2.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
    a2.set_xticks(range(3))
    a2.set_xticklabels([n for _, n in names], fontsize=9)
    a2.set_ylabel("Median latency per request (s)", color=t["ink2"], fontsize=10)
    a2.set_title("Measured on the GPU (left bar 7.5%, right 5%)", color=t["ink"], fontsize=10.5, loc="left")
    f.tight_layout()
    f.savefig(out / f"escalation_{theme}.png", facecolor=t["surface"], bbox_inches="tight")
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tr = json.load(open(RES / "transfer.json"))
    lat = json.load(open(RES / "escalation_results.json"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        fig(theme, t, tr, lat, out)
    print("wrote", sorted(p.name for p in out.glob("escalation_*.png")))


if __name__ == "__main__":
    main()
