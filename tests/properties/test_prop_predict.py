"""Properties of ``KrigingModel.predict`` that hold for every sample set.

Covers ``_model.KrigingModel.predict`` and ``_targets.execute_on`` /
``_targets._execute_grid``: the grid path, the point path, chunking by
``max_memory`` and the ``where`` mask.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv


@pytest.mark.parametrize('dim', [2, 3])
def test_zero_nugget_kriging_returns_the_samples(cases, seed, dim):
    """At a sample, exact kriging returns the sample value with zero variance."""
    samples = cases.samples(seed, dim)
    result = cases.fit(samples, seed).predict(samples)
    scale = np.ptp(samples['v']) or 1.0
    np.testing.assert_allclose(result['v'], samples['v'], atol=1e-6 * scale, err_msg=f'{seed=}')
    assert np.abs(result['v variance']).max() <= 1e-6 * scale**2, f'{seed=}'


@pytest.mark.parametrize('dim', [2, 3])
def test_grid_path_matches_point_path(cases, seed, dim):
    """An ImageData target gives the same values as its points kriged one by one."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, dim)
    on_grid = model.predict(grid)
    on_points = model.predict(pv.PolyData(grid.points))
    for name in ('v', 'v variance'):
        np.testing.assert_allclose(
            on_grid[name], on_points[name], rtol=1e-7, atol=1e-9, err_msg=f'{seed=} {name}'
        )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('target', ['grid', 'points'])
def test_chunking_does_not_change_the_result(cases, seed, dim, target):
    """Splitting the work by max_memory gives the same values as one call."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, dim)
    mesh = grid if target == 'grid' else pv.PolyData(grid.points)
    # One kriging-matrix column per sample plus the Lagrange row, 8 bytes each.
    per_point = (samples.n_points + 1) * 8
    points_per_chunk = int(np.random.default_rng(seed).integers(1, grid.n_points))
    chunked = model.predict(mesh, max_memory=points_per_chunk * per_point)
    whole = model.predict(mesh, max_memory=None)
    for name in ('v', 'v variance'):
        np.testing.assert_allclose(
            chunked[name], whole[name], rtol=1e-7, atol=1e-9, err_msg=f'{seed=} {name}'
        )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('target', ['grid', 'points'])
def test_where_keeps_selected_points_and_nulls_the_rest(cases, seed, dim, target):
    """``where`` changes nothing at selected points and writes null_value elsewhere."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, dim)
    mesh = grid if target == 'grid' else pv.PolyData(grid.points)
    where = np.random.default_rng(seed).random(mesh.n_points) < 0.5
    masked = model.predict(mesh, where=where, null_value=-999.0)
    whole = model.predict(mesh)
    for name in ('v', 'v variance'):
        np.testing.assert_allclose(
            masked[name][where], whole[name][where], rtol=1e-7, atol=1e-9, err_msg=f'{seed=}'
        )
        assert np.all(masked[name][~where] == -999.0), f'{seed=} {name}'
    np.testing.assert_array_equal(masked['vtkValidPointMask'], where.astype(np.uint8))
