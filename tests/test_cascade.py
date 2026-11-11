import numpy as np
import pytest

from uniinfer.controller import Cascade, best_static_config, fit_cascade, simulate


def _toy(n=400, seed=0, informative=True):
    """Config 0 is expensive and always right; config 1 is cheap and right only on 'easy' requests.
    The confidence of config 1 is high on easy requests (if informative) or random noise."""
    rng = np.random.default_rng(seed)
    easy = rng.random(n) < 0.7
    S = np.c_[np.ones(n), easy.astype(float)]
    C = np.tile([1.0, 0.2], (n, 1))
    q1 = np.where(easy, 0.9, 0.3) + 0.05 * rng.normal(size=n) if informative else rng.random(n)
    Q = np.c_[np.ones(n), q1]
    return S, C, Q, easy


def test_simulate_two_stage_sum_cost():
    S = np.array([[1.0, 0.0], [1.0, 1.0]])
    C = np.array([[1.0, 0.2], [1.0, 0.2]])
    Q = np.array([[1.0, 0.1], [1.0, 0.9]])
    sc, co, st = simulate(Cascade((1, 0), (0.5,)), S, C, Q)
    assert sc.tolist() == [1.0, 1.0]
    assert co.tolist() == pytest.approx([1.2, 0.2])  # escalated request pays for both attempts
    assert st.tolist() == [1, 0]
    _, co_max, _ = simulate(Cascade((1, 0), (0.5,)), S, C, Q, cost_mode="max")
    assert co_max.tolist() == pytest.approx([1.0, 0.2])


def test_single_stage_is_static():
    S, C, Q, _ = _toy()
    sc, co, _ = simulate(Cascade((1,), ()), S, C, Q)
    assert np.allclose(sc, S[:, 1]) and np.allclose(co, C[:, 1])


def test_informative_confidence_beats_static():
    S, C, Q, _ = _toy()
    B = 0.6
    cas, train_score, train_cost = fit_cascade(S, C, Q, B)
    assert train_cost <= B + 1e-9
    static = S[:, best_static_config(S, C, B)].mean()
    assert cas.stages == (1, 0)
    assert train_score > static + 0.2


def test_uninformative_confidence_gives_no_free_lunch():
    S, C, Q, easy = _toy(informative=False)
    B = 0.6
    _, train_score, _ = fit_cascade(S, C, Q, B)
    # Random escalation: the best feasible mix is about 0.7 + 0.3 * P(escalate). It cannot approach 1.0.
    assert train_score < 0.9


def test_three_stage_search_runs_and_respects_budget():
    rng = np.random.default_rng(1)
    S = rng.random((100, 4))
    C = np.tile([1.0, 0.6, 0.3, 0.1], (100, 1))
    Q = rng.random((100, 4))
    cas, _, cost = fit_cascade(S, C, Q, 0.5, max_stages=3, n_thresholds=7)
    assert cost <= 0.5 + 1e-9 and 1 <= len(cas.stages) <= 3
