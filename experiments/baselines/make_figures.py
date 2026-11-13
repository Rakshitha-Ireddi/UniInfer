"""Figures for the v0.2 baselines study (light and dark), from the merged analysis (70-config grid + low SnapKV
ratios) in results/v0.2-baselines-low/baselines_results.json.

    python experiments/baselines/make_figures.py
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

RES = REPO / "results" / "v0.2-baselines-low"
OUT = REPO / "docs" / "figures"
DATASETS = {"textvqa": "TextVQA", "docvqa": "DocVQA", "mmbench": "MMBench"}


def _legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def fig_operators(theme, t, res, out):
    """One panel per dataset: score vs keep ratio for every single-axis operator."""
    ops = res["operators"]
    lines = [("visual", "uniform", "Visual: uniform", t["series"][0], "-"),
             ("visual", "dissim", "Visual: redundancy (dissim)", t["series"][1], "-"),
             ("visual", "attn", "Visual: FastV-style (attention)", t["series"][2], "-"),
             ("KV", "l2norm", "KV: L2-norm", t["series"][3], (0, (4, 2))),
             ("KV", "snapkv", "KV: SnapKV", t["ink2"], (0, (4, 2)))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=160, facecolor=t["surface"], sharey=True)
    for ax, (d, label) in zip(axes, DATASETS.items()):
        style(ax, t)
        for axis, m, name, col, ls in lines:
            pts = [(float(k.split("keep")[1]), v[d]) for k, v in ops.items()
                   if d in v and k.split()[0] == axis and k.split()[1] == m]
            if not pts:
                continue
            x, y = zip(*sorted(pts + [(1.0, res["uncompressed"][d])]))
            ax.plot(x, y, color=col, lw=2, ls=ls, marker="o", ms=5, mec=t["surface"], mew=1.2, label=name, zorder=3)
        ax.set_title(label, color=t["ink"], fontsize=10.5, loc="left")
        ax.set_xlabel("Fraction of tokens kept", color=t["ink2"], fontsize=9.5)
        ax.invert_xaxis()
    axes[0].set_ylabel("Score (luck-free)", color=t["ink2"], fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
                     bbox_to_anchor=(0.5, -0.01))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.suptitle("Compression operators on their own: attention-guided methods vs simple ones",
                 color=t["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"operators_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def fig_headroom(theme, t, res, out):
    """Static vs oracle per budget, for the v0.1 operator family and for all operators."""
    hr = res["headroom"]
    fams = [("v0.1 operators (uniform/dissim + L2-norm)", "v0.1 operators", t["series"][0]),
            ("all operators", "+ SnapKV + FastV-style", t["series"][2])]
    budgets = sorted(next(iter(hr.values())), key=float, reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160, facecolor=t["surface"])
    style(ax, t)
    x = list(range(len(budgets)))
    for k, (fam, label, col) in enumerate(fams):
        off = (k - 0.5) * 0.36
        ok = [i for i, b in zip(x, budgets) if not hr[fam][b].get("infeasible")]
        st = [hr[fam][budgets[i]]["static"] for i in ok]
        orc = [hr[fam][budgets[i]]["oracle"] for i in ok]
        ax.bar([i + off for i in ok], st, width=0.34, color=col, alpha=0.9, label=f"Best static, {label}", zorder=3)
        ax.scatter([i + off for i in ok], orc, marker="_", s=420, color=col, lw=2.5, zorder=4,
                   label=f"Per-request oracle, {label}")
        for i, s, o in zip(ok, st, orc):
            ax.annotate(f"+{o - s:.2f}", (i + off, o), xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=7.5, color=t["ink2"])
        for i in sorted(set(x) - set(ok)):
            ax.annotate("not\nreachable",(i + off, 0.0), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, color=t["ink2"], xycoords=("data", "axes fraction"))
    ax.set_xticks(x, [f"{float(b) * 100:g}%" for b in budgets])
    ax.set_xlabel("Average KV-memory budget", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Score (luck-free)", color=t["ink2"], fontsize=10)
    lo = min(hr[f][b]["static"] for f, _, _ in fams for b in budgets if not hr[f][b].get("infeasible"))
    ax.set_ylim(min(0.3, lo - 0.05), 0.95)
    ax.set_title("Does the oracle headroom survive stronger operators? (labels: oracle minus static)",
                 color=t["ink"], fontsize=11, loc="left", pad=10)
    _legend(ax, t, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"headroom_operators_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def fig_allocation(theme, t, res, out):
    """Held-out gain of task-aware allocation over the best static config (paired 95% CI), per budget."""
    al = res["allocation"]
    fams = [("v0.1 operators (uniform/dissim + L2-norm)", "v0.1 operators", t["series"][0], -0.09),
            ("all operators", "+ SnapKV + FastV-style", t["series"][2], 0.09)]
    budgets = sorted({b for f, *_ in fams for b in al[f]["task-aware minus static"]}, key=float, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=160, facecolor=t["surface"])
    style(ax, t)
    ax.axhline(0, color=t["ink2"], lw=1, zorder=2)
    for fam, label, col, off in fams:
        d = al[fam]["task-aware minus static"]
        xs = [i for i, b in enumerate(budgets) if b in d]
        mid = [d[budgets[i]]["diff"] for i in xs]
        err = [[m - d[budgets[i]]["low"] for i, m in zip(xs, mid)],
               [d[budgets[i]]["high"] - m for i, m in zip(xs, mid)]]
        ax.errorbar([i + off for i in xs], mid, yerr=err, fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.6,
                    capsize=3, mec=t["surface"], mew=1.2, label=label, zorder=3)
    ax.set_xticks(range(len(budgets)), [f"{float(b) * 100:g}%" for b in budgets])
    ax.set_xlabel("Average KV-memory budget", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Task-aware minus best static", color=t["ink2"], fontsize=10)
    ax.set_title("Held-out gain of task-aware allocation (paired 95% CI)", color=t["ink"], fontsize=11, loc="left",
                 pad=10)
    _legend(ax, t, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"allocation_operators_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RES / "baselines_results.json"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res = json.load(open(a.results))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        fig_operators(theme, t, res, out)
        fig_headroom(theme, t, res, out)
        fig_allocation(theme, t, res, out)
    print("wrote", sorted(p.name for p in out.glob("*operators*.png")))


if __name__ == "__main__":
    main()
