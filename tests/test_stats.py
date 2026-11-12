import numpy as np
import pytest

from uniinfer.stats import (
    best_static_replicates,
    bootstrap_ci,
    bootstrap_weights,
    oracle_replicates,
    paired_diff,
)


def test_weights_resample_rows_and_respect_strata():
    groups = np.array([0] * 30 + [1] * 70)
    W = bootstrap_weights(100, 500, seed=1, groups=groups)
    assert W.shape == (500, 100)
    assert np.all(W[:, :30].sum(1) == 30) and np.all(W[:, 30:].sum(1) == 70)
    U = bootstrap_weights(100, 500, seed=1)
    assert np.all(U.sum(1) == 100)


def test_deterministic_with_seed():
    x = np.random.default_rng(0).random(200)
    assert bootstrap_ci(x, seed=7) == bootstrap_ci(x, seed=7)
    assert bootstrap_ci(x, seed=7) != bootstrap_ci(x, seed=8)


def test_ci_covers_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, big = rng.random(50), rng.random(5000)
    lo_s, hi_s = bootstrap_ci(small)
    lo_b, hi_b = bootstrap_ci(big)
    assert lo_s < small.mean() < hi_s and lo_b < big.mean() < hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s) / 5


def test_constant_values_give_zero_width():
    assert bootstrap_ci(np.ones(40)) == (pytest.approx(1.0), pytest.approx(1.0))


def test_paired_diff_detects_real_and_null_differences():
    rng = np.random.default_rng(0)
    base = rng.random(300)
    better = paired_diff(base + 0.05, base)
    assert better["low"] > 0 and better["p_le_0"] == 0.0
    noise = paired_diff(base + rng.normal(0, 0.1, 300), base)
    assert noise["low"] < 0 < noise["high"]


def test_static_and_oracle_replicates_match_point_estimates_on_identity_weights():
    rng = np.random.default_rng(2)
    S = (rng.random((120, 6)) > 0.3).astype(float)
    C = np.tile(np.linspace(1, 0.1, 6), (120, 1))
    W = np.ones((1, 120))  # one "replicate" = the original sample
    feasible = C.mean(0) <= 0.5
    assert best_static_replicates(S, C, 0.5, W)[0] == pytest.approx(S.mean(0)[feasible].max())
    orc = oracle_replicates(S, C, 0.5, W)[0]
    assert orc >= S.mean(0)[feasible].max() - 1e-12  # oracle is never worse than static


def test_static_is_reselected_per_replicate():
    """Two configs with equal cost; which is 'best' depends on the resample, so re-selection must
    give replicates >= each fixed config's replicate."""
    rng = np.random.default_rng(3)
    S = rng.random((80, 2))
    C = np.full((80, 2), 0.3)
    W = bootstrap_weights(80, 300, seed=0)
    reps = best_static_replicates(S, C, 0.5, W)
    fixed = W @ S / 80
    assert np.all(reps >= fixed.max(1) - 1e-12)
