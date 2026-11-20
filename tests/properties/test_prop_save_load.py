"""Properties of ``KrigingModel.save`` and ``load``, issue #7. Skips until it lands."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista_krige

pytestmark = pytest.mark.skipif(
    not hasattr(pyvista_krige.KrigingModel, 'load'),
    reason='KrigingModel.save and load are not implemented yet (issue #7)',
)


@pytest.mark.parametrize('dim', [2, 3])
def test_a_loaded_model_predicts_exactly_like_the_saved_one(cases, seed, dim, tmp_path):
    samples = cases.samples(seed, dim)
    model = cases.fit(samples, seed)
    path = tmp_path / 'model.npz'
    model.save(path)
    loaded = pyvista_krige.KrigingModel.load(path)
    grid = cases.grid(samples, seed, dim)
    before, after = model.predict(grid), loaded.predict(grid)
    np.testing.assert_array_equal(after['v'], before['v'], err_msg=f'{seed=}')
    np.testing.assert_array_equal(after['v variance'], before['v variance'], err_msg=f'{seed=}')
