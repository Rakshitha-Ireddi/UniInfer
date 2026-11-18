"""Figure for the v0.3 scale-up (light and dark), from results/v0.3-scale/scale_results.json.

    python experiments/scale/figures.py          # docs/figures/scale_snapkv_{light,dark}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from common import REPO  # noqa: E402
from make_figures import THEMES, style  # noqa: E402

RES = REPO / "results" / "v0.3-scale"
OUT = REPO / "docs" / "figures"
DATASETS = {"docvqa": "DocVQA", "textvqa": "TextVQA", "chartqa": "ChartQA (new)", "mmbench": "MMBench"}


def fig_scale(theme, t, res, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=160, facecolor=t["surface"],
                                 gridspec_kw=dict(width_ratios=[1.1, 1]))
    style(a1, t)
    cols = t["series"]
    for (d, label), col in zip(DATASETS.items(), cols):
        pts = sorted((float(r), v[d]) for r, v in res["curves"].items())
        x, y = zip(*pts)
        a1.plot(x, y, color=col, lw=2, marker="o", ms=4.5, mec=t["surface"], mew=1.1, label=label, zorder=3)
    a1.set_xscale("log")
    a1.set_xticks([0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0])
    a1.set_xticklabels(["2%", "3%", "5%", "10%", "25%", "50%", "100%"])
    a1.minorticks_off()
    a1.invert_xaxis()
    a1.set_xlabel("KV cache kept by SnapKV (log scale)", color=t["ink2"], fontsize=9.5)
    a1.set_ylabel("Score (luck-free)", color=t["ink2"], fontsize=10)
    a1.set_title(f"SnapKV per task, n = {res['n']} requests", color=t["ink"], fontsize=10.5, loc="left")
    leg = a1.legend(frameon=False, fontsize=8.5, loc="lower left")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    style(a2, t)
    comps = [("task-aware - static", "Task-aware − static", cols[0], -0.12),
             ("cascade - shuffled", "Cascade − shuffled confidence", cols[1], 0.12)]
    budgets = list(res["paired"]["task-aware - static"])
    xs = range(len(budgets))
    for key, label, col, off in comps:
        d = [res["paired"][key][b] for b in budgets]
        y = [v["diff"] for v in d]
        err = [[v["diff"] - v["low"] for v in d], [v["high"] - v["diff"] for v in d]]
        a2.errorbar([x + off for x in xs], y, yerr=err, fmt="o", color=col, ms=5, capsize=3, lw=1.6, label=label,
                    zorder=3)
    a2.axhline(0, color=t["muted"], lw=1)
    a2.set_xticks(list(xs))
    a2.set_xticklabels([f"{float(b) * 100:g}%" for b in budgets])
    a2.set_xlabel("Average KV budget", color=t["ink2"], fontsize=9.5)
    a2.set_ylabel("Held-out score difference (95% CI)", color=t["ink2"], fontsize=10)
    a2.set_title("Held-out gains (5-fold × 3 seeds)", color=t["ink"], fontsize=10.5, loc="left")
    leg = a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(out / f"scale_snapkv_{theme}.png", facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RES / "scale_results.json"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res = json.load(open(a.results))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        fig_scale(theme, t, res, out)
    print("wrote", sorted(p.name for p in out.glob("scale_snapkv_*.png")))


if __name__ == "__main__":
    main()
