"""Properties of ``KrigingModel.simulate``, issue #9. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige

pytestmark = pytest.mark.skipif(
    not hasattr(pyvista_krige.KrigingModel, 'simulate'),
    reason='simulate is not implemented yet (issue #9)',
)


def _realizations(result: pv.DataSet) -> list[np.ndarray]:
    names = sorted(n for n in result.point_data if n.startswith('v realization '))
    return [np.asarray(result[name]) for name in names]


@pytest.mark.parametrize('dim', [2, 3])
def test_every_realization_honors_the_samples(cases, seed, dim):
    """With no nugget, each realization equals the data at the sample locations."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    target = pv.PolyData(np.vstack([samples.points, cases.grid(samples, seed, dim).points]))
    runs = _realizations(model.simulate(target, n=3, seed=seed))
    assert len(runs) == 3, f'{seed=}'
    scale = np.ptp(samples['v']) or 1.0
    for run in runs:
        np.testing.assert_allclose(
            run[: samples.n_points], samples['v'], atol=1e-5 * scale, err_msg=f'{seed=}'
        )


def test_the_same_seed_gives_the_same_realizations(cases, seed):
    samples = cases.samples(seed, 2)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, 2)
    first = _realizations(model.simulate(grid, n=2, seed=seed))
    again = _realizations(model.simulate(grid, n=2, seed=seed))
    other = _realizations(model.simulate(grid, n=2, seed=seed + 1))
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.allclose(a, b) for a, b in zip(first, other, strict=True)), f'{seed=}'
