"""``predict(preference='cell')``: kriging at the cell centers onto ``cell_data``."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige


@pytest.fixture
def model3d(samples3d):
    return samples3d.krige.fit('temperature', variogram_model='spherical')


@pytest.fixture
def blocks() -> pv.ImageData:
    """A block model covering the 3D samples."""
    return pv.ImageData(dimensions=(6, 5, 4), spacing=(2.0, 2.5, 1.5), origin=(0.0, 0.0, 0.2))


def at_centers(model, mesh, **kwargs):
    return model.predict(pv.PolyData(mesh.cell_centers().points), **kwargs)


def _targets(blocks: pv.ImageData) -> dict[str, pv.DataSet]:
    rectilinear = pv.RectilinearGrid(
        np.array([0.0, 1.0, 3.0, 6.0, 10.0]), np.array([0.0, 4.0, 10.0]), np.array([0.0, 2.0, 5.0])
    )
    return {
        'image': blocks,
        'rectilinear': rectilinear,
        'structured': blocks.cast_to_structured_grid(),
        'unstructured': blocks.cast_to_unstructured_grid(),
        'polydata': blocks.extract_surface(algorithm=None),
    }


@pytest.mark.parametrize(
    'kind', ['image', 'rectilinear', 'structured', 'unstructured', 'polydata']
)
def test_cell_values_equal_kriging_at_the_cell_centers(model3d, blocks, kind):
    mesh = _targets(blocks)[kind]
    result = model3d.predict(mesh, preference='cell')
    expected = at_centers(model3d, mesh)
    assert result.n_cells == mesh.n_cells
    for name in ('temperature', 'temperature variance'):
        assert name not in result.point_data
        np.testing.assert_allclose(result.cell_data[name], expected[name], rtol=1e-10, atol=1e-12)


def test_point_preference_is_the_default(model3d, blocks):
    default = model3d.predict(blocks)
    explicit = model3d.predict(blocks, preference='point')
    np.testing.assert_array_equal(default['temperature'], explicit['temperature'])
    assert 'temperature' in default.point_data
    assert 'temperature' not in default.cell_data


def test_axis_aligned_grid_keeps_the_grid_path(model3d, blocks, monkeypatch):
    calls = []
    execute = model3d._instance.execute

    def spy(style, *args, **kwargs):
        calls.append(style)
        return execute(style, *args, **kwargs)

    monkeypatch.setattr(model3d._instance, 'execute', spy)
    model3d.predict(blocks, preference='cell')
    model3d.predict(blocks, preference='cell', where=np.arange(blocks.n_cells) % 2 == 0)
    assert calls == ['grid', 'masked']


def test_2d_model_on_a_single_layer_image(samples2d):
    model = samples2d.krige.fit('elevation')
    image = pv.ImageData(dimensions=(8, 6, 1), spacing=(1.5, 2.0, 1.0))
    result = model.predict(image, preference='cell', std=True)
    expected = at_centers(model, image, std=True)
    for name in ('elevation', 'elevation std'):
        np.testing.assert_allclose(result.cell_data[name], expected[name], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('chunked', [False, True])
def test_where_reads_cell_data_and_nulls_the_other_cells(model3d, blocks, chunked):
    mesh = blocks.copy()
    keep = mesh.cell_centers().points[:, 0] < 5
    mesh.cell_data['inside'] = keep
    max_memory = 8 * (model3d._samples.n_points + 1) * 7 if chunked else 1e9
    result = model3d.predict(mesh, preference='cell', where='inside', max_memory=max_memory)
    expected = at_centers(model3d, mesh)
    values = result.cell_data['temperature']
    np.testing.assert_allclose(values[keep], expected['temperature'][keep], rtol=1e-10)
    assert np.isnan(values[~keep]).all()
    np.testing.assert_array_equal(result.cell_data['vtkValidCellMask'], keep.astype(np.uint8))
    assert 'vtkValidPointMask' not in result.point_data


def test_where_array_must_have_one_value_per_cell(model3d, blocks):
    with pytest.raises(ValueError, match=r'one value per target cell \(60\)'):
        model3d.predict(blocks, preference='cell', where=np.ones(blocks.n_points, dtype=bool))


def test_max_distance_and_max_std_apply_to_the_cell_centers(model3d):
    far = pv.ImageData(dimensions=(9, 3, 3), spacing=(5.0, 5.0, 2.5), origin=(-10.0, 0.0, 0.0))
    centers = pv.PolyData(far.cell_centers().points)
    std = model3d.predict(centers, std=True)['temperature std']
    for kwargs in ({'max_distance': 2.0}, {'max_std': float(np.median(std))}):
        on_cells = model3d.predict(far, preference='cell', **kwargs)
        expected = model3d.predict(centers, **kwargs)
        nulled = np.isnan(on_cells.cell_data['temperature'])
        assert 0 < nulled.sum() < far.n_cells
        np.testing.assert_array_equal(nulled, np.isnan(expected['temperature']))
        np.testing.assert_array_equal(
            on_cells.cell_data['vtkValidCellMask'], expected['vtkValidPointMask']
        )


def test_drift_arrays_read_cell_data(samples2d):
    samples = samples2d.copy()
    samples['depth'] = samples.points[:, 0]
    model = samples.krige.fit(
        'elevation', method='universal', variogram_model='linear', specified_drift=['depth']
    )
    mesh = pv.ImageData(dimensions=(7, 6, 1), spacing=(1.5, 2.0, 1.0))
    mesh.cell_data['depth'] = mesh.cell_centers().points[:, 0]
    centers = pv.PolyData(mesh.cell_centers().points)
    centers['depth'] = mesh.cell_data['depth']
    expected = model.predict(centers)
    for max_memory in (1e9, 3 * 50 * 8):
        result = model.predict(mesh, preference='cell', max_memory=max_memory)
        np.testing.assert_allclose(
            result.cell_data['elevation'], expected['elevation'], rtol=1e-9, atol=1e-12
        )


def test_transformed_arrays_go_on_cells_and_exceedance_reads_them(blocks):
    rng = np.random.default_rng(3)
    samples = pv.PolyData(rng.uniform((0, 0, 0), (10, 10, 5), (40, 3)))
    samples['grade'] = np.exp(samples.points[:, 0] / 4 + rng.normal(scale=0.2, size=40))
    model = samples.krige.fit('grade', transform='log', variogram_model='spherical')
    result = model.predict(blocks, preference='cell', std=True)
    expected = at_centers(model, blocks, std=True)
    for name in ('grade', 'grade std', 'grade transformed', 'grade transformed std'):
        assert name not in result.point_data
        np.testing.assert_allclose(result.cell_data[name], expected[name], rtol=1e-10)
    probability = pyvista_krige.exceedance(result, 'grade', 3.0, model=model)
    reference = pyvista_krige.exceedance(expected, 'grade', 3.0, model=model)
    np.testing.assert_allclose(probability.cell_data['P(grade > 3)'], reference['P(grade > 3)'])
    with pytest.raises(ValueError, match='Pass that model'):
        pyvista_krige.exceedance(result, 'grade', 3.0)


def test_exceedance_and_plot_helpers_work_on_a_cell_result(model3d, blocks):
    result = model3d.predict(blocks, preference='cell')
    probability = pyvista_krige.exceedance(result, 'temperature', 5.0)
    values = probability.cell_data['P(temperature > 5)']
    assert values.shape == (blocks.n_cells,)
    assert ((values >= 0) & (values <= 1)).all()
    pl = pv.Plotter()
    pl.krige.add_uncertainty(result, 'temperature')
    pl.krige.add_exceedance(result, 'temperature', 5.0)
    pl.close()


def test_to_image_forwards_the_preference(samples3d):
    image = samples3d.krige.to_image('temperature', dimensions=(5, 5, 4), preference='cell')
    assert image.cell_data['temperature'].shape == (image.n_cells,)
    assert 'temperature' not in image.point_data


@pytest.mark.parametrize('preference', ['cells', 'field', None])
def test_rejects_other_preferences(model3d, blocks, preference):
    with pytest.raises(ValueError, match="preference must be 'point' or 'cell'"):
        model3d.predict(blocks, preference=preference)
