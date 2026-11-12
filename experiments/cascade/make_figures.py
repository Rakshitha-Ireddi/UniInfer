"""Render the v0.2 cascade figures (light and dark) from results/v0.2-cascade/cascade_results.json.

    python experiments/cascade/make_figures.py
"""
from __future__ import annotations

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

V2 = REPO / "results" / "v0.2-cascade"
OUT = REPO / "docs" / "figures"
DATASET_LABELS = {"textvqa": "TextVQA", "docvqa": "DocVQA", "mmbench": "MMBench"}


def legend(ax, t, fontsize=9, **kw):
    leg = ax.legend(frameon=False, fontsize=fontsize, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def fig_frontier(theme, t, res):
    """Two panels, one per cost model: real confidence vs the shuffled-confidence control."""
    table, best = res["table"], res["best_signal"]
    panels = [("", "Failed attempts are paid for (sum cost)"),
              (" | max-cost", "Memory freed before a retry (max cost)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160, facecolor=t["surface"], sharey=True)
    for ax, (suffix, title) in zip(axes, panels):
        style(ax, t)
        rows = [("static", "Static: one config for all", t["series"][0], "-", 2),
                ("task-aware", "Task-aware allocation (v0.1 best)", t["series"][1], "-", 2),
                (f"task-aware cascade | {best} SHUFFLED{suffix}" if suffix else
                 f"task-aware cascade | {best} SHUFFLED (control)",
                 "Cascade, confidence SHUFFLED (control)", t["muted"], (0, (1, 1.5)), 2),
                (f"task-aware cascade | {best}{suffix}", "Cascade, real confidence", t["series"][2], "-", 2.4),
                ("oracle (luck-free)", "Per-request oracle (upper bound)", t["muted"], (0, (4, 3)), 1.5)]
        for k, label, col, ls, lw in rows:
            if k not in table:
                continue
            x, y = zip(*sorted((v["cost"], v["score"]) for v in table[k].values()))
            ax.plot(x, y, color=col, lw=lw, ls=ls, marker="o", ms=6, mec=t["surface"], mew=1.5, label=label,
                    zorder=3)
        ax.set_ylim(0.5, 0.9)
        ax.set_title(title, color=t["ink"], fontsize=10.5, loc="left", pad=8)
        ax.set_xlabel("Average KV memory actually used", color=t["ink2"], fontsize=10)
    axes[0].set_ylabel("Task score (held-out)", color=t["ink2"], fontsize=10)
    legend(axes[1], t, loc="lower right", fontsize=8)
    fig.suptitle("Confidence only pays off when a retry does not double the memory bill",
                 color=t["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"cascade_frontier_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def fig_auroc(theme, t, res):
    auc = res["auroc"]
    sigs = sorted(auc, key=lambda s: auc[s]["all"])
    groups = ["all", *[d for d in DATASET_LABELS if d in next(iter(auc.values()))]]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160, facecolor=t["surface"])
    style(ax, t)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.grid(False, axis="y")
    h = 0.8 / len(groups)
    for g_i, (g, col) in enumerate(zip(groups, [t["ink2"], *t["series"]])):
        ys = [s_i + (g_i - (len(groups) - 1) / 2) * h for s_i in range(len(sigs))]
        ax.barh(ys, [auc[s][g] for s in sigs], height=h * 0.9, color=col,
                label="All requests" if g == "all" else DATASET_LABELS[g], zorder=3)
    ax.axvline(0.5, color=t["muted"], lw=1, ls=(0, (2, 2)), zorder=4)
    ax.text(0.505, len(sigs) - 0.45, "chance", color=t["ink2"], fontsize=9)
    ax.set_yticks(range(len(sigs)), sigs)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC: does the signal tell kept answers from damaged ones?", color=t["ink2"], fontsize=10)
    ax.set_title("Model confidence under compression is a per-request damage detector",
                 color=t["ink"], fontsize=11, loc="left", pad=12)
    legend(ax, t, loc="upper left", bbox_to_anchor=(1.01, 1.0), reverse=True)
    fig.tight_layout()
    fig.savefig(OUT / f"cascade_auroc_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    res = json.load(open(V2 / "cascade_results.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        fig_frontier(theme, t, res)
        fig_auroc(theme, t, res)
    print("wrote", sorted(p.name for p in OUT.glob("cascade_*.png")))


if __name__ == "__main__":
    main()
