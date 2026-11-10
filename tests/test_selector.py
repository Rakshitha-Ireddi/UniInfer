import numpy as np
import pytest

from uniinfer.controller import BudgetedSelector, best_static_config, select_under_lambda, tune_lambda


def _toy(n=200, seed=0):
    """Two request types: 'easy' ones lose nothing when compressed; 'hard' ones need the full config."""
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.5
    C = np.tile(np.array([1.0, 0.5, 0.1]), (n, 1))
    S = np.where(hard[:, None], np.array([1.0, 0.3, 0.0]), np.array([1.0, 1.0, 1.0]))
    X = np.c_[hard.astype(float), rng.normal(size=n)]
    return X, S, C, hard


def test_static_picks_best_feasible():
    X, S, C, _ = _toy()
    j = best_static_config(S, C, 0.5)
    assert C[:, j].mean() <= 0.5 and j == 1


def test_lambda_meets_budget():
    _, S, C, _ = _toy()
    lam = tune_lambda(S, C, 0.55)
    j = select_under_lambda(S, C, lam)
    assert C[np.arange(len(C)), j].mean() <= 0.55 + 1e-9


def test_learned_selector_beats_static_when_signal_exists():
    X, S, C, hard = _toy()
    sel = BudgetedSelector(np.array([[1.0], [0.5], [0.1]]), max_iter=50).fit(X[:150], S[:150], C[:150])
    j = sel.select(X[150:], C[150:], 0.55)
    learned = S[150:][np.arange(50), j].mean()
    static = S[150:, best_static_config(S[:150], C[:150], 0.55)].mean()
    assert learned > static + 0.1
    assert (j[hard[150:]] == 0).mean() > 0.8, "hard requests should get the full config"


def test_select_requires_fit():
    sel = BudgetedSelector(np.zeros((3, 1)))
    with pytest.raises(AssertionError):
        sel.lambda_for(0.5)
