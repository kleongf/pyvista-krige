"""Properties of ``pyvista_krige.exceedance`` (``_exceedance.exceedance``)."""

from __future__ import annotations

import numpy as np

import pyvista_krige


def _result(cases, seed):
    samples = cases.samples(seed, 2)
    grid = cases.grid(samples, seed, 2)
    return cases.fit(samples, seed).predict(grid)


def test_probability_is_bounded_and_falls_with_the_threshold(cases, seed):
    """P(v > t) lies in [0, 1] and never rises as t rises."""
    result = _result(cases, seed)
    low, high = np.sort(np.random.default_rng(seed).normal(size=2))
    p_low = pyvista_krige.exceedance(result, 'v', low, name='p')['p']
    p_high = pyvista_krige.exceedance(result, 'v', high, name='p')['p']
    assert np.all((p_low >= 0) & (p_low <= 1)), f'{seed=}'
    assert np.all(p_high <= p_low + 1e-12), f'{seed=} {low=} {high=}'


def test_probability_is_the_normal_tail_of_the_estimate(cases, seed):
    """P(v > t) is the tail of N(estimate, kriging variance) above t, point by point.

    Bounds, monotonicity and the complement all survive a miscalibrated
    probability (say, every value scaled by 1.01); only this catches it.
    """
    from scipy.stats import norm

    result = _result(cases, seed)
    mean, sigma = result['v'], np.sqrt(np.clip(result['v variance'], 0, None))
    threshold = float(np.random.default_rng(seed).choice(mean) + 0.3 * sigma.max())
    p = pyvista_krige.exceedance(result, 'v', threshold, name='p')['p']
    uncertain = sigma > 1e-6 * sigma.max()
    expected = norm.sf((threshold - mean[uncertain]) / sigma[uncertain])
    np.testing.assert_allclose(p[uncertain], expected, rtol=1e-9, atol=1e-12, err_msg=f'{seed=}')
    np.testing.assert_array_equal(p[~uncertain], (mean[~uncertain] > threshold).astype(float))


def test_above_and_below_add_up_to_one(cases, seed):
    """P(v > t) + P(v < t) = 1 at every point."""
    result = _result(cases, seed)
    threshold = float(np.random.default_rng(seed).normal())
    above = pyvista_krige.exceedance(result, 'v', threshold, name='p')['p']
    below = pyvista_krige.exceedance(result, 'v', threshold, below=True, name='p')['p']
    np.testing.assert_allclose(above + below, 1.0, atol=1e-12, err_msg=f'{seed=}')


def test_std_and_variance_inputs_agree(cases, seed):
    """Passing the std array gives the same probability as the variance array."""
    samples = cases.samples(seed, 2)
    grid = cases.grid(samples, seed, 2)
    model = cases.fit(samples, seed)
    threshold = float(np.random.default_rng(seed).normal())
    from_var = pyvista_krige.exceedance(model.predict(grid), 'v', threshold, name='p')['p']
    from_std = pyvista_krige.exceedance(model.predict(grid, std=True), 'v', threshold, name='p')
    np.testing.assert_allclose(from_std['p'], from_var, atol=1e-9, err_msg=f'{seed=}')
