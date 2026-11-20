"""Seeded random cases for the property suite.

Every test that takes a ``seed`` argument runs once per seed. Raise the count
with ``KRIGE_PROPERTY_CASES`` (default 8) for a deeper run. A failure names its
seed, and ``random_samples(seed, dim)`` rebuilds the exact case.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige

N_CASES = int(os.environ.get('KRIGE_PROPERTY_CASES', '8'))

# Zero-nugget variograms keep kriging exact at the samples. Gaussian is left out
# because with no nugget its kriging matrix is too ill-conditioned to test tightly.
_MODELS = ('spherical', 'exponential', 'linear')


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if 'seed' in metafunc.fixturenames:
        metafunc.parametrize('seed', range(N_CASES))


def random_samples(seed: int, dim: int, *, positive: bool = False) -> pv.PolyData:
    """Return 8 to 30 samples on a jittered lattice with a random field ``'v'``.

    The lattice keeps samples apart so the kriging matrix stays well conditioned,
    and each axis gets its own random scale so the cases are not all square.
    """
    rng = np.random.default_rng(seed)
    per_axis = rng.integers(3, 6 if dim == 2 else 4, size=dim)
    lattice = np.stack(np.meshgrid(*(np.arange(k) for k in per_axis)), -1).reshape(-1, dim)
    lattice = lattice[rng.permutation(len(lattice))[: rng.integers(8, 31)]]
    coords = (lattice + rng.uniform(-0.3, 0.3, lattice.shape)) * rng.uniform(0.5, 20.0, dim)
    coords += rng.uniform(-100, 100, dim)
    points = np.zeros((len(coords), 3))
    points[:, :dim] = coords
    samples = pv.PolyData(points)
    trend = coords @ rng.normal(size=dim) / np.ptp(coords, axis=0).max()
    values = trend + rng.normal(scale=0.5, size=len(coords))
    samples['v'] = np.exp(values) if positive else values
    return samples


def fit_exact(samples: pv.PolyData, seed: int, **kwargs: object) -> pyvista_krige.KrigingModel:
    """Fit ordinary kriging with a random zero-nugget variogram, no fitting noise."""
    rng = np.random.default_rng(seed + 10_000)
    model = _MODELS[seed % len(_MODELS)]
    extent = float(np.ptp(samples.points, axis=0).max())
    sill = float(np.var(samples['v'])) or 1.0
    if model == 'linear':
        parameters = {'slope': sill / extent, 'nugget': 0.0}
    else:
        parameters = {'sill': sill, 'range': rng.uniform(0.3, 1.5) * extent, 'nugget': 0.0}
    dim = 2 if np.ptp(samples.points[:, 2]) == 0 else 3
    return samples.krige.fit(
        'v', dim=dim, variogram_model=model, variogram_parameters=parameters, **kwargs
    )


def grid_over(samples: pv.PolyData, seed: int, dim: int) -> pv.ImageData:
    """Return a small axis-aligned grid covering the samples, padded 10 percent."""
    rng = np.random.default_rng(seed + 20_000)
    lower, upper = samples.points.min(axis=0), samples.points.max(axis=0)
    pad = 0.1 * (upper - lower)
    dims = np.ones(3, dtype=int)
    dims[:dim] = rng.integers(3, 9 if dim == 2 else 6, size=dim)
    spacing = np.divide(upper - lower + 2 * pad, dims - 1, out=np.ones(3), where=dims > 1)
    return pv.ImageData(dimensions=tuple(dims), spacing=tuple(spacing), origin=tuple(lower - pad))


def accepts_fit_option(name: str, value: object) -> bool:
    """Whether ``fit`` handles ``name`` itself instead of passing it on to PyKrige.

    Until a planned feature lands, PyKrige rejects the unknown keyword, and the
    feature's property module skips.
    """
    samples = random_samples(0, 2, positive=True)
    try:
        samples.krige.fit('v', **{name: value})
    except TypeError as error:
        return 'unexpected keyword argument' not in str(error)
    return True


@pytest.fixture
def cases():
    """The case builders, so test modules need no import from this conftest."""

    class Cases:
        samples = staticmethod(random_samples)
        fit = staticmethod(fit_exact)
        grid = staticmethod(grid_over)
        accepts = staticmethod(accepts_fit_option)

    return Cases
