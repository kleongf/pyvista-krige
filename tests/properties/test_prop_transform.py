"""Properties of ``fit(transform=...)``, issue #2. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest

# The normal score back-transform integrates by an 81-node rule, about 0.2 % of
# the data range off a fine reference at 100 seeds. These allow 3x that.
TOL_MEAN = 5e-3
TOL_VAR = 3e-3


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


@pytest.mark.parametrize('dim', [2, 3])
def test_log_results_are_the_lognormal_moments(cases, seed, dim):
    """Away from the samples, the log estimate and variance are the lognormal mean and variance.

    The sample checks above see zero kriging variance, so they pass even if the
    back-transform ignores the variance; this sees the whole grid.
    """
    samples = cases.samples(seed, dim, positive=True)
    result = cases.fit(samples, seed, transform='log').predict(cases.grid(samples, seed, dim))
    log_mean = result['v transformed']
    log_variance = np.clip(result['v transformed variance'], 0, None)
    mean = np.exp(log_mean + log_variance / 2)
    np.testing.assert_allclose(result['v'], mean, rtol=1e-9, err_msg=f'{seed=}')
    np.testing.assert_allclose(
        result['v variance'], mean**2 * np.expm1(log_variance), rtol=1e-9, err_msg=f'{seed=}'
    )


@pytest.mark.parametrize('dim', [2, 3])
def test_normal_score_results_are_the_moments_of_the_score_distribution(cases, seed, dim):
    """The normal score estimate and variance are those of the back-transformed score.

    Checked against a much finer integration over the model's own table, so the
    variance in score space has to carry into the original units.
    """
    from scipy.special import ndtr

    samples = cases.samples(seed, dim, positive=True)
    model = cases.fit(samples, seed, transform='normal_score')
    result = model.predict(cases.grid(samples, seed, dim))
    table = model.transform
    score = result['v transformed'][:, None]
    std = np.sqrt(np.clip(result['v transformed variance'], 0, None))[:, None]
    nodes = np.linspace(-8, 8, 4001)
    weights = np.exp(-(nodes**2) / 2)
    weights /= weights.sum()
    values = np.interp(ndtr(score + std * nodes), table.cdf, table.values)
    mean = values @ weights
    variance = (values**2) @ weights - mean**2
    spread = np.ptp(samples['v'])
    np.testing.assert_allclose(result['v'], mean, atol=TOL_MEAN * spread, err_msg=f'{seed=}')
    np.testing.assert_allclose(
        result['v variance'], variance, atol=TOL_VAR * spread**2, err_msg=f'{seed=}'
    )
