"""Properties of ``KrigingModel.suggest_samples``, issue #4. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige

pytestmark = pytest.mark.skipif(
    not hasattr(pyvista_krige.KrigingModel, 'suggest_samples'),
    reason='suggest_samples is not implemented yet (issue #4)',
)


def _std_with(samples, extra_points, model, dim, target):
    """Kriging std on ``target`` after adding ``extra_points``, variogram held fixed.

    Kriging variance depends only on where the samples are, so the added points
    can carry any value.
    """
    points = np.vstack([samples.points, extra_points])
    augmented = pv.PolyData(points)
    augmented['v'] = np.concatenate([samples['v'], np.zeros(len(extra_points))])
    refit = augmented.krige.fit(
        'v',
        dim=dim,
        variogram_model=model.variogram.model_name,
        variogram_parameters=list(model.variogram.parameters),
    )
    return refit.predict(target, std=True)['v std']


@pytest.mark.parametrize('dim', [2, 3])
def test_each_pick_has_the_largest_std_given_the_earlier_picks(cases, seed, dim):
    """Greedy picking: pick k maximizes the std after picks 0..k-1 are added."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, dim)
    n = int(np.random.default_rng(seed).integers(1, 5))
    picks = model.suggest_samples(grid, n=n)
    assert picks.n_points == n, f'{seed=}'
    assert len(np.unique(picks.points, axis=0)) == n, f'{seed=}: repeated pick'
    for k in range(n):
        std = _std_with(samples, picks.points[:k], model, dim, grid)
        match = np.flatnonzero(np.all(np.isclose(grid.points, picks.points[k]), axis=1))
        assert match.size == 1, f'{seed=}: pick {k} is not a target point'
        assert std[match[0]] >= std.max() - 1e-9, f'{seed=}: pick {k} is not the max std'
