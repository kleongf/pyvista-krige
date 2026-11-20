"""Round-trip properties of the ESRI ASCII and ZMAP readers and writers, issue #8.

Skips until they land.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige

FORMATS = {
    'esri_ascii': ('write_esri_ascii', 'read_esri_ascii', 'grid.asc'),
    'zmap': ('write_zmap', 'read_zmap', 'grid.zmap'),
}


def _random_grid(seed: int) -> pv.ImageData:
    """A single-layer grid with non-square cells, a random origin and some NaN nodes."""
    rng = np.random.default_rng(seed)
    nx, ny = rng.integers(2, 25, size=2)
    grid = pv.ImageData(
        dimensions=(int(nx), int(ny), 1),
        spacing=(*rng.uniform(0.1, 50.0, size=2), 1.0),
        origin=(*rng.uniform(-1e5, 1e5, size=2), 0.0),
    )
    values = rng.normal(scale=10 ** rng.uniform(-2, 4), size=grid.n_points)
    values[rng.random(grid.n_points) < 0.15] = np.nan
    grid['z'] = values
    return grid


@pytest.mark.parametrize('fmt', FORMATS)
def test_write_then_read_returns_the_same_grid(seed, fmt, tmp_path):
    write_name, read_name, filename = FORMATS[fmt]
    if not hasattr(pyvista_krige, read_name):
        pytest.skip(f'{read_name} is not implemented yet (issue #8)')
    grid = _random_grid(seed)
    path = tmp_path / filename
    getattr(pyvista_krige, write_name)(grid, path, scalars='z')
    back = getattr(pyvista_krige, read_name)(path)
    assert back.dimensions == grid.dimensions, f'{seed=}'
    scale = np.nanmax(np.abs(grid['z']))
    np.testing.assert_allclose(back.origin[:2], grid.origin[:2], rtol=1e-6, err_msg=f'{seed=}')
    np.testing.assert_allclose(back.spacing[:2], grid.spacing[:2], rtol=1e-6, err_msg=f'{seed=}')
    values = back.point_data[back.point_data.keys()[0]]
    np.testing.assert_array_equal(np.isnan(values), np.isnan(grid['z']), err_msg=f'{seed=}')
    finite = ~np.isnan(grid['z'])
    np.testing.assert_allclose(
        values[finite], grid['z'][finite], rtol=1e-6, atol=1e-6 * scale, err_msg=f'{seed=}'
    )
