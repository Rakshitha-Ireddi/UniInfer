"""CPU helpers for uncertainty estimates from independent observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the mean, resampling observations with replacement.

    A local generator makes the result reproducible without changing global RNG state.
    Batches bound temporary memory when there are many observations.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty, finite, one-dimensional sequence")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
