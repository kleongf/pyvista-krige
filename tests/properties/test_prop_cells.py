"""Properties of ``predict(preference='cell')``, issue #3. Skips until it lands."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige

pytestmark = pytest.mark.skipif(
    'preference' not in inspect.signature(pyvista_krige.KrigingModel.predict).parameters,
    reason="predict(preference='cell') is not implemented yet (issue #3)",
)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('target', ['grid', 'unstructured'])
def test_cell_result_equals_kriging_at_cell_centers(cases, seed, dim, target):
    """Cell values are the point estimates at the cell centers, cell for cell."""
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    grid = cases.grid(samples, seed, dim)
    mesh = grid if target == 'grid' else grid.cast_to_unstructured_grid()
    on_cells = model.predict(mesh, preference='cell')
    at_centers = model.predict(pv.PolyData(mesh.cell_centers().points))
    for name in ('v', 'v variance'):
        assert name in on_cells.cell_data, f'{name!r} missing from cell_data'
        np.testing.assert_allclose(
            on_cells.cell_data[name], at_centers[name], rtol=1e-7, atol=1e-9, err_msg=f'{seed=}'
        )
