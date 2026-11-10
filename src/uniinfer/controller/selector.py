"""Budgeted configuration selection.

Problem: N requests, K candidate configs. Request i under config c has quality S[i,c] and cost C[i,c]
(KV memory relative to uncompressed). Maximize mean quality subject to mean cost <= B.

The per-request oracle uses true S. A learned selector replaces S with a prediction S_hat from request
features, then solves the Lagrangian relaxation  argmax_c S_hat[i,c] - lambda * C[i,c]  per request.
lambda is tuned on training data so the average cost meets B (ADR-0005).
"""
from __future__ import annotations

import numpy as np

LAMBDAS = np.concatenate([[0.0], np.logspace(-3, 1.5, 120)])


def select_under_lambda(P: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    """Per-row argmax of P - lam*C. Ties go to the cheaper config."""
    return (P - lam * C - 1e-6 * C).argmax(1)


def tune_lambda(P: np.ndarray, C: np.ndarray, budget: float) -> float:
    """Smallest lambda on the grid whose selections meet the average-cost budget."""
    for lam in LAMBDAS:
        j = select_under_lambda(P, C, lam)
        if C[np.arange(len(C)), j].mean() <= budget + 1e-9:
            return float(lam)
    return float(LAMBDAS[-1])


def best_static_config(S: np.ndarray, C: np.ndarray, budget: float) -> int:
    """Index of the single config with the best mean quality among those meeting the budget."""
    feasible = C.mean(0) <= budget + 1e-9
    if not feasible.any():
        return int(C.mean(0).argmin())
    return int(np.where(feasible, S.mean(0), -np.inf).argmax())


def config_pair_features(X: np.ndarray, config_desc: np.ndarray, C: np.ndarray) -> np.ndarray:
    """One row per (request, config): request features, config descriptor, realized cost."""
    N, K = C.shape
    return np.concatenate([np.repeat(X, K, 0), np.tile(config_desc, (N, 1)), C.reshape(-1, 1)], 1)


class BudgetedSelector:
    """Gradient-boosted quality predictor plus Lagrangian budgeted selection."""

    def __init__(self, config_desc: np.ndarray, seed: int = 0, **gbm_kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
        params.update(gbm_kwargs)
        self.desc = config_desc
        self.model = HistGradientBoostingRegressor(random_state=seed, **params)
        self._train_pred: np.ndarray | None = None
        self._train_cost: np.ndarray | None = None

    def fit(self, X: np.ndarray, S: np.ndarray, C: np.ndarray) -> BudgetedSelector:
        """Train the quality predictor once; budgets are handled at selection time."""
        self.model.fit(config_pair_features(X, self.desc, C), S.reshape(-1))
        self._train_pred, self._train_cost = self.predict_quality(X, C), C
        return self

    def predict_quality(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        return self.model.predict(config_pair_features(X, self.desc, C)).reshape(C.shape)

    def lambda_for(self, budget: float) -> float:
        """Lagrange multiplier that meets `budget` on the training requests."""
        assert self._train_pred is not None, "call fit() first"
        return tune_lambda(self._train_pred, self._train_cost, budget)

    def select(self, X: np.ndarray, C: np.ndarray, budget: float) -> np.ndarray:
        return select_under_lambda(self.predict_quality(X, C), C, self.lambda_for(budget))
