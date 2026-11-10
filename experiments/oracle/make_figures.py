"""Render the README figures (light and dark variants) from results/v0.1-oracle.

    python experiments/oracle/make_figures.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import BUDGETS, REPO, RESULTS, load_sweep, oracle_score

OUT = REPO / "docs" / "figures"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["muted"])
    ax.tick_params(colors=t["ink2"], labelsize=10)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def fig_budget(theme, t, pred, full, oracle_pts):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160, facecolor=t["surface"])
    style(ax, t)
    series = [("static", "Static: one config for all"),
              ("task label only", "Task-aware allocation"),
              ("T2 +question emb", "Learned per-request selector (T2)"),
              ("oracle", "Per-request oracle (luck-free upper bound)")]
    short = {"static": "Static", "task label only": "Task-aware", "T2 +question emb": "Learned (T2)",
             "oracle": "Oracle"}
    for (k, label), col in zip(series, t["series"]):
        pts = oracle_pts if k == "oracle" else sorted((v["cost"], v["score"]) for v in pred[k].values())
        x, y = zip(*pts)
        ls = (0, (4, 3)) if k == "oracle" else "-"
        ax.plot(x, y, color=col, lw=2, ls=ls, marker="o", ms=7, mec=t["surface"], mew=2, label=label, zorder=3)
        dy = {"T2 +question emb": -9, "task label only": 7}.get(k, 0)
        ax.annotate(short[k], (x[-1], y[-1]), xytext=(9, dy if k != "oracle" else 9), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=t["ink2"])
    ax.axhline(full, color=t["muted"], lw=1, ls=(0, (2, 2)), zorder=2)
    ax.text(0.01, full + 0.008, f"uncompressed model ({full:.3f})", color=t["ink2"], fontsize=9, ha="left")
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("Average KV-cache memory used (fraction of uncompressed)", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Task score (held-out)", color=t["ink2"], fontsize=10)
    ax.set_title("Quality vs memory: Qwen2.5-VL-3B on TextVQA + DocVQA + MMBench (450 requests)",
                 color=t["ink"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(OUT / f"quality_vs_memory_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def fig_sensitivity(theme, t, S, C, ds):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=t["surface"])
    style(ax, t)
    names = {"mmbench": "MMBench (general)", "textvqa": "TextVQA (scene text)", "docvqa": "DocVQA (documents)"}
    for (d, label), col in zip(names.items(), t["series"]):
        idx = np.where(ds == d)[0]
        cheapest = np.sort([C[i, S[i] >= S[i, 0] - 1e-9].min() for i in idx])
        y = np.arange(1, len(cheapest) + 1) / len(cheapest)
        k = np.searchsorted(cheapest, 0.25, side="right")
        ax.step(np.r_[0, cheapest], np.r_[0, y], where="post", color=col, lw=2,
                label=f"{label}: {k / len(cheapest):.0%} fine at 25% memory")
    ax.axvline(0.25, color=t["muted"], lw=1, ls=(0, (2, 2)))
    ax.text(0.255, 0.04, "25% memory", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Smallest KV memory that keeps the uncompressed answer quality", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Fraction of requests", color=t["ink2"], fontsize=10)
    ax.set_title("Requests differ: share of requests that survive at each memory level",
                 color=t["ink"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(OUT / f"request_sensitivity_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, S, C, ds = load_sweep()
    pred = json.load(open(RESULTS / "predictor_results.json"))
    Sc = np.minimum(S, S[:, :1])  # luck-free: no credit for beating the uncompressed model
    oracle_pts = sorted((B, oracle_score(Sc, C, B)) for B in BUDGETS)
    for theme, t in THEMES.items():
        fig_budget(theme, t, pred, S[:, 0].mean(), oracle_pts)
        fig_sensitivity(theme, t, S, C, ds)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
