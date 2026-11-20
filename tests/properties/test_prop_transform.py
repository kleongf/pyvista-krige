"""Properties of ``fit(transform=...)``, issue #2. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _needs_transform(cases):
    if not cases.accepts('transform', 'log'):
        pytest.skip('fit(transform=...) is not implemented yet (issue #2)')


@pytest.mark.parametrize('transform', ['log', 'normal_score'])
@pytest.mark.parametrize('dim', [2, 3])
def test_back_transform_returns_the_samples(cases, seed, dim, transform):
    """With no nugget the back-transformed estimate at a sample is that sample."""
    samples = cases.samples(seed, dim, positive=True)
    result = cases.fit(samples, seed, transform=transform).predict(samples)
    np.testing.assert_allclose(result['v'], samples['v'], rtol=1e-6, err_msg=f'{seed=}')
    assert np.abs(result['v variance']).max() <= 1e-6 * np.ptp(samples['v']) ** 2, f'{seed=}'


@pytest.mark.parametrize('dim', [2, 3])
def test_log_transform_gives_positive_estimates(cases, seed, dim):
    """A lognormal back-transform never returns a zero or negative value."""
    samples = cases.samples(seed, dim, positive=True)
    result = cases.fit(samples, seed, transform='log').predict(cases.grid(samples, seed, dim))
    assert np.all(result['v'] > 0), f'{seed=}'
    assert np.all(result['v variance'] >= 0), f'{seed=}'


def test_no_transform_matches_the_default(cases, seed):
    """``transform=None`` is exactly the untransformed model."""
    samples = cases.samples(seed, 2)
    grid = cases.grid(samples, seed, 2)
    default = cases.fit(samples, seed).predict(grid)
    explicit = cases.fit(samples, seed, transform=None).predict(grid)
    np.testing.assert_array_equal(explicit['v'], default['v'])
    np.testing.assert_array_equal(explicit['v variance'], default['v variance'])
