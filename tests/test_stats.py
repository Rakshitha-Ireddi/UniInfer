import numpy as np
import pytest

from uniinfer.stats import bootstrap_ci


def test_bootstrap_known_interval():
    # The bootstrap means for two observations are 0, 0.5 and 1.
    assert bootstrap_ci([0, 1]) == (0.0, 1.0)
    assert bootstrap_ci([3, 3, 3]) == (3.0, 3.0)
    assert bootstrap_ci([7]) == (7.0, 7.0)


def test_bootstrap_is_deterministic_and_matches_seeded_resampling():
    values = np.arange(7, dtype=float)
    rng = np.random.default_rng(13)
    means = values[rng.integers(len(values), size=(300, len(values)))].mean(1)
    expected = np.quantile(means, [0.1, 0.9])
    actual = bootstrap_ci(values, n=300, alpha=0.2, seed=13)
    np.testing.assert_allclose(actual, expected)
    assert actual == bootstrap_ci(values, n=300, alpha=0.2, seed=13)


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [np.inf]])
def test_bootstrap_rejects_invalid_observations(values):
    with pytest.raises(ValueError):
        bootstrap_ci(values)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 1.5}, {"alpha": 0}, {"alpha": 1}])
def test_bootstrap_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], **kwargs)
