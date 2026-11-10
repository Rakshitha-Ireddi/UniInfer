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
BUDGETS = [0.5, 0.375, 0.25, 0.15]


def config_key(c: dict) -> str:
    return f"{c['v_method']}@{c['v_ratio']:.2f}|kv@{c['kv_ratio']:.2f}"


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
    return np.array([[c["v_ratio"], c["kv_ratio"], c["v_ratio"] * c["kv_ratio"],
                      float(c["v_method"] == "uniform"), float(c["v_method"] == "dissim")]
                     for c in recs[0]["configs"]], np.float32)


def oracle_score(S: np.ndarray, C: np.ndarray, budget: float) -> float:
    """Per-request oracle under an AVERAGE cost budget (Lagrangian sweep over true scores)."""
    best = float("nan")
    for lam in np.concatenate([[0], np.logspace(-4, 2, 400)]):
        j = (S - lam * C - 1e-9 * C).argmax(1)
        if C[np.arange(len(S)), j].mean() <= budget + 1e-9:
            sc = float(S[np.arange(len(S)), j].mean())
            best = sc if np.isnan(best) else max(best, sc)
    return best
