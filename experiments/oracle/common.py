"""Shared helpers for the oracle experiment: load sweep results into matrices."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("UNIINFER_RESULTS", REPO / "results" / "v0.1-oracle"))

V_RATIOS = [1.0, 0.75, 0.5, 0.25]
V_METHODS = ["uniform", "dissim"]
KV_RATIOS = [1.0, 0.75, 0.5, 0.25]
KV_METHODS = ["l2norm"]
BUDGETS = [0.5, 0.375, 0.25, 0.15]


def config_key(c: dict) -> str:
    """e.g. 'dissim@0.50|kv@0.25'. Non-default KV methods are named: 'attn@0.50|snapkv@0.25'."""
    km = c.get("kv_method", "l2norm")
    kv = "kv" if km == "l2norm" or c["kv_ratio"] >= 1.0 else km
    return f"{c['v_method']}@{c['v_ratio']:.2f}|{kv}@{c['kv_ratio']:.2f}"


def load_sweep(path: str | Path = RESULTS / "oracle_sweep.jsonl"):
    """Return (records, config keys, S quality (N,K), C relative KV cost (N,K), dataset names (N,))."""
    recs = [json.loads(line) for line in open(path) if line.strip()]
    keys = [config_key(c) for c in recs[0]["configs"]]
    S = np.array([[c["score"] for c in r["configs"]] for r in recs])
    full_kv = np.array([r["configs"][0]["kv_len"] for r in recs], float)
    C = np.array([[c["kv_len"] for c in r["configs"]] for r in recs]) / full_kv[:, None]
    ds = np.array([r["dataset"] for r in recs])
    return recs, keys, S, C, ds


def config_descriptors(recs) -> np.ndarray:
    """Per-config features for the selector. Method flags for attention-guided methods are added only when the
    sweep contains them, so v0.1/v0.2 descriptors (and results) are unchanged."""
    cfgs = recs[0]["configs"]
    extra_v = [m for m in ("attn",) if any(c["v_method"] == m for c in cfgs)]
    extra_k = [m for m in ("snapkv",) if any(c.get("kv_method") == m for c in cfgs)]
    return np.array([[c["v_ratio"], c["kv_ratio"], c["v_ratio"] * c["kv_ratio"],
                      float(c["v_method"] == "uniform"), float(c["v_method"] == "dissim")]
                     + [float(c["v_method"] == m) for m in extra_v]
                     + [float(c.get("kv_method", "l2norm") == m and c["kv_ratio"] < 1.0) for m in extra_k]
                     for c in cfgs], np.float32)


def oracle_score(S: np.ndarray, C: np.ndarray, budget: float) -> float:
    """Per-request oracle under an AVERAGE cost budget (Lagrangian sweep over true scores)."""
    best = float("nan")
    for lam in np.concatenate([[0], np.logspace(-4, 2, 400)]):
        j = (S - lam * C - 1e-9 * C).argmax(1)
        if C[np.arange(len(S)), j].mean() <= budget + 1e-9:
            sc = float(S[np.arange(len(S)), j].mean())
            best = sc if np.isnan(best) else max(best, sc)
    return best
