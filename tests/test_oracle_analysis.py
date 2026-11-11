import numpy as np
import pytest
from analyze import bootstrap_score_ci, per_task_score, report
from common import BUDGETS, oracle_score

from uniinfer.controller import best_static_config


@pytest.mark.parametrize("tasks", [["a", "a", "a", "a"], ["a", "a", "b", "b"]])
def test_bootstrap_reselects_static_and_per_task(tasks):
    S = np.tile([[1.0, 0.0], [0.0, 1.0]], (2, 1))
    C = np.full_like(S, 0.5)
    ci = bootstrap_score_ci(S, C, np.array(tasks), [0.5], n=400)
    # A fixed full-sample winner can score zero; reselecting always scores >= 0.5.
    np.testing.assert_allclose(ci[0], [[0.5, 1.0], [0.5, 1.0], [1.0, 1.0]])


def test_bootstrap_oracle_uses_average_budget_and_retunes_each_replicate():
    S = np.array([[1.0, 0.0], [0.0, 0.0]])
    C = np.array([[1.0, 0.1], [1.0, 0.1]])
    ci = bootstrap_score_ci(S, C, np.array(["a", "a"]), [0.55], n=200)
    # Mixed draws can spend 1.0 on the first row and 0.1 on the second. Duplicating
    # the first row makes that policy infeasible, so its oracle score must fall to 0.
    np.testing.assert_allclose(ci[0], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.5]])


def test_cached_bootstrap_matches_direct_row_resampling_and_is_deterministic():
    S = np.array([[1.0, 0.2, 0.0], [1.0, 0.7, 0.1], [0.6, 0.5, 0.0], [0.8, 0.2, 0.1]])
    C = np.array([[1.0, 0.5, 0.1], [1.0, 0.2, 0.1], [1.0, 0.4, 0.1], [1.0, 0.8, 0.1]])
    ds = np.array(["a", "a", "b", "b"])
    budgets = [0.35, 0.6]
    n, seed = 24, 13
    samples = np.random.default_rng(seed).integers(len(S), size=(n, len(S)))
    expected = []
    for budget in budgets:
        scores = []
        for indices in samples:
            s, c, tasks = S[indices], C[indices], ds[indices]
            scores.append(
                [
                    s[:, best_static_config(s, c, budget)].mean(),
                    per_task_score(s, c, tasks, budget),
                    oracle_score(s, c, budget),
                ]
            )
        expected.append(np.quantile(scores, [0.025, 0.975], axis=0).T)
    actual = bootstrap_score_ci(S, C, ds, budgets, n=n, seed=seed)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(actual, bootstrap_score_ci(S, C, ds, budgets, n=n, seed=seed))


@pytest.mark.parametrize("data_seed, shape", [(7, (9, 4)), (28, (7, 5))])
def test_bootstrap_cost_ties_match_direct_resampling(data_seed, shape):
    rng = np.random.default_rng(data_seed)
    S = rng.random(shape)
    # Discrete costs create ties, including in the cheapest-config fallback.
    C = rng.choice([0.1, 0.2, 0.3], size=shape)
    ds = np.arange(shape[0]) % 3
    budgets = [0.05, 0.2]
    # Check individual replicates so quantiles cannot hide a changed selection.
    for seed in range(32):
        indices = np.random.default_rng(seed).integers(len(S), size=(1, len(S)))[0]
        s, c, tasks = S[indices], C[indices], ds[indices]
        expected = np.array(
            [
                [
                    s[:, best_static_config(s, c, budget)].mean(),
                    per_task_score(s, c, tasks, budget),
                    oracle_score(s, c, budget),
                ]
                for budget in budgets
            ]
        )
        actual = bootstrap_score_ci(S, C, ds, budgets, n=1, seed=seed)
        for bound in range(2):
            np.testing.assert_array_equal(actual[:, :2, bound], expected[:, :2])
            np.testing.assert_allclose(actual[:, 2, bound], expected[:, 2], atol=1e-12, equal_nan=True)


def test_report_puts_score_and_ci_in_the_same_cell(capsys):
    S, C = np.ones((2, 1)), np.full((2, 1), 0.1)
    ds = np.array(["a", "a"])
    intervals = np.tile([0.8, 1.0], (len(BUDGETS), 3, 1))
    report(S, C, ds, ["config"], "Luck-free", intervals)
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.count("1.000 [0.800, 1.000]") for line in lines) == 3 * len(BUDGETS)
    assert not any("95% CI:" in line for line in lines)
    report(S, C, ds, ["config"], "Raw scores")
    lines = capsys.readouterr().out.splitlines()
    assert lines[2] == f"  {'budget':>6s}  {'static':>7s}  {'per-task':>8s}  {'oracle':>7s}  {'gap':>6s}  static config"
    assert lines[3] == f"  {BUDGETS[0]:6.3f}  {1.0:7.3f}  {1.0:8.3f}  {1.0:7.3f}  {0.0:+6.3f}  config"


def test_infeasible_oracle_is_not_reported_as_feasible():
    ci = bootstrap_score_ci(np.ones((2, 1)), np.ones((2, 1)), np.array(["a", "b"]), [0.1], n=8)
    assert np.isnan(ci[0, 2]).all()
    # Preserve best_static_config's cheapest-config fallback when none is feasible.
    np.testing.assert_array_equal(ci[0, :2], np.ones((2, 2)))
