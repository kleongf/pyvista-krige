"""Properties of ``samples.krige.decluster``, issue #6. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from pyvista_krige import KrigeAccessor

pytestmark = pytest.mark.skipif(
    not hasattr(KrigeAccessor, 'decluster'),
    reason='decluster is not implemented yet (issue #6)',
)

WEIGHT = 'declustering weight'


@pytest.mark.parametrize('dim', [2, 3])
def test_weights_are_positive_and_sum_to_n(cases, seed, dim):
    samples = cases.samples(seed, dim)
    size = float(np.random.default_rng(seed).uniform(0.05, 0.5)) * np.ptp(samples.points).max()
    weights = samples.krige.decluster(cell_size=size)[WEIGHT]
    assert np.all(weights > 0), f'{seed=}'
    np.testing.assert_allclose(weights.sum(), samples.n_points, rtol=1e-9, err_msg=f'{seed=}')


@pytest.mark.parametrize('dim', [2, 3])
def test_weights_follow_the_samples_not_their_order(cases, seed, dim):
    """Shuffling the samples shuffles the weights the same way."""
    samples = cases.samples(seed, dim)
    size = 0.2 * np.ptp(samples.points).max()
    order = np.random.default_rng(seed).permutation(samples.n_points)
    shuffled = pv.PolyData(samples.points[order])
    shuffled['v'] = samples['v'][order]
    np.testing.assert_allclose(
        shuffled.krige.decluster(cell_size=size)[WEIGHT],
        samples.krige.decluster(cell_size=size)[WEIGHT][order],
        rtol=1e-9,
        err_msg=f'{seed=}',
    )


@pytest.mark.parametrize('dim', [2, 3])
def test_a_regular_lattice_gets_equal_weights(seed, dim):
    """With the cell size equal to the spacing, every cell holds one sample at any offset."""
    rng = np.random.default_rng(seed)
    spacing = float(rng.uniform(0.5, 10.0))
    counts = rng.integers(3, 7, size=dim)
    lattice = np.stack(np.meshgrid(*(np.arange(k) for k in counts)), -1).reshape(-1, dim)
    points = np.zeros((len(lattice), 3))
    points[:, :dim] = lattice * spacing
    samples = pv.PolyData(points)
    samples['v'] = rng.normal(size=len(points))
    weights = samples.krige.decluster(cell_size=spacing)[WEIGHT]
    np.testing.assert_allclose(weights, 1.0, rtol=1e-9, err_msg=f'{seed=}')
