from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from pykrige.ok import OrdinaryKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk import UniversalKriging
from pykrige.uk3d import UniversalKriging3D


def test_predict_at_samples_reproduces_sample_values(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    out = model.predict(samples2d)
    np.testing.assert_allclose(out['elevation'], samples2d['elevation'], atol=1e-8)


def test_image_target_matches_pykrige_point_estimates(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    image = pv.ImageData(dimensions=(7, 5, 1), spacing=(1.5, 2.0, 1.0), origin=(0.5, 0.0, 0.0))
    out = model.predict(image)
    expected, _ = model.instance.execute('points', image.points[:, 0], image.points[:, 1])
    assert isinstance(out, pv.ImageData)
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_rectilinear_target_matches_pykrige_point_estimates(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    grid = pv.RectilinearGrid([0.0, 1.0, 4.0, 9.0], [2.0, 3.0, 7.5])
    out = model.predict(grid)
    expected, _ = model.instance.execute('points', grid.points[:, 0], grid.points[:, 1])
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_predict_writes_variance_zero_at_samples_positive_elsewhere(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    at_samples = model.predict(samples2d)
    far_point = model.predict(pv.PolyData([[30.0, 30.0, 0.0]]))
    np.testing.assert_allclose(at_samples['elevation variance'], 0.0, atol=1e-8)
    assert far_point['elevation variance'][0] > 0.0


def test_3d_samples_match_independent_ordinary_kriging_3d(samples3d):
    model = samples3d.krige.fit('temperature', variogram_model='linear')
    volume = pv.ImageData(dimensions=(4, 3, 3), spacing=(3.0, 4.0, 2.0))
    out = model.predict(volume)
    oracle = OrdinaryKriging3D(
        *samples3d.points.T, samples3d['temperature'], variogram_model='linear'
    )
    expected, _ = oracle.execute('points', *volume.points.T)
    np.testing.assert_allclose(out['temperature'], np.ma.getdata(expected))


def test_universal_2d_matches_independent_universal_kriging(samples2d):
    model = samples2d.krige.fit(
        'elevation', method='universal', variogram_model='linear', drift_terms=['regional_linear']
    )
    target = pv.PolyData([[1.0, 2.0, 0.0], [5.5, 5.5, 0.0], [12.0, -1.0, 0.0]])
    out = model.predict(target)
    oracle = UniversalKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='linear',
        drift_terms=['regional_linear'],
    )
    expected, _ = oracle.execute('points', target.points[:, 0], target.points[:, 1])
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_universal_3d_matches_independent_universal_kriging_3d(samples3d):
    model = samples3d.krige.fit(
        'temperature',
        method='universal',
        variogram_model='linear',
        drift_terms=['regional_linear'],
    )
    volume = pv.ImageData(dimensions=(3, 3, 2), spacing=(4.0, 4.0, 4.0))
    out = model.predict(volume)
    oracle = UniversalKriging3D(
        *samples3d.points.T,
        samples3d['temperature'],
        variogram_model='linear',
        drift_terms=['regional_linear'],
    )
    expected, _ = oracle.execute('points', *volume.points.T)
    np.testing.assert_allclose(out['temperature'], np.ma.getdata(expected))


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'method': 'simple'}, "method must be 'ordinary' or 'universal'"),
        ({'dim': 4}, 'dim must be 2 or 3'),
    ],
)
def test_fit_rejects_unknown_method_or_dim(samples2d, kwargs, match):
    with pytest.raises(ValueError, match=match):
        samples2d.krige.fit('elevation', **kwargs)


def test_predict_rejects_multiblock_with_combine_hint(samples2d):
    model = samples2d.krige.fit('elevation')
    blocks = pv.MultiBlock([pv.Plane()])
    with pytest.raises(TypeError, match=r'combine\(\)'):
        model.predict(blocks)


def test_cell_scalars_are_sampled_at_cell_centers():
    cells = pv.ImageData(dimensions=(6, 5, 1), spacing=(2.0, 2.0, 1.0))
    cells.cell_data['porosity'] = np.linspace(0.1, 0.3, cells.n_cells)
    model = cells.krige.fit('porosity', variogram_model='linear')
    target = pv.PolyData([[3.3, 4.1, 0.0], [7.0, 1.5, 0.0]])
    out = model.predict(target)
    centers = cells.cell_centers().points
    oracle = OrdinaryKriging(
        centers[:, 0], centers[:, 1], cells.cell_data['porosity'], variogram_model='linear'
    )
    expected, _ = oracle.execute('points', target.points[:, 0], target.points[:, 1])
    np.testing.assert_allclose(out['porosity'], np.ma.getdata(expected))


def test_nan_samples_are_dropped_with_warning(samples2d):
    values = samples2d['elevation'].copy()
    values[[3, 17]] = np.nan
    samples2d['elevation'] = values
    with pytest.warns(UserWarning, match='Dropping 2 samples'):
        model = samples2d.krige.fit('elevation', variogram_model='linear')
    target = pv.PolyData([[4.0, 4.0, 0.0]])
    out = model.predict(target)
    keep = np.isfinite(values)
    oracle = OrdinaryKriging(
        samples2d.points[keep, 0],
        samples2d.points[keep, 1],
        values[keep],
        variogram_model='linear',
    )
    expected, _ = oracle.execute('points', [4.0], [4.0])
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_n_closest_points_matches_pykrige_moving_window(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    image = pv.ImageData(dimensions=(6, 6, 1), spacing=(2.0, 2.0, 1.0))
    out = model.predict(image, n_closest_points=8)
    oracle = OrdinaryKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='spherical',
    )
    expected, _ = oracle.execute(
        'points', image.points[:, 0], image.points[:, 1], backend='loop', n_closest_points=8
    )
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


@pytest.mark.parametrize(
    ('fit_kwargs', 'predict_kwargs', 'match'),
    [
        ({'method': 'universal'}, {'n_closest_points': 8}, 'not supported for universal'),
        ({'dim': 3}, {'backend': 'C'}, "backend 'C' is only available for 2D ordinary"),
        ({}, {'n_closest_points': 1}, 'n_closest_points must be at least 2'),
        ({}, {'backend': 'fast'}, "backend must be one of 'vectorized', 'loop', 'C'"),
    ],
)
def test_predict_rejects_unsupported_backend_options(samples3d, fit_kwargs, predict_kwargs, match):
    model = samples3d.krige.fit('temperature', variogram_model='linear', **fit_kwargs)
    with pytest.raises(ValueError, match=match):
        model.predict(pv.PolyData([[1.0, 1.0, 1.0]]), **predict_kwargs)


@pytest.mark.parametrize('fixture', ['samples2d', 'samples3d'])
@pytest.mark.parametrize('kind', ['image', 'points', 'masked'])
def test_predict_chunks_to_stay_under_max_memory(request, fixture, kind):
    samples = request.getfixturevalue(fixture)
    scalars = samples.array_names[0]
    model = samples.krige.fit(scalars, variogram_model='spherical')
    target = pv.ImageData(dimensions=(9, 8, 1 if fixture == 'samples2d' else 6))
    where = None
    if kind == 'points':
        target = target.cast_to_pointset()
    elif kind == 'masked':
        where = target.points[:, 0] < 4.5
    whole = model.predict(target, max_memory=None, where=where)
    # A budget of a few points' rows forces many chunks.
    chunked = model.predict(target, max_memory=5 * (model.n_samples + 1) * 8, where=where)
    np.testing.assert_allclose(chunked[scalars], whole[scalars], equal_nan=True)
    np.testing.assert_allclose(
        chunked[f'{scalars} variance'], whole[f'{scalars} variance'], equal_nan=True
    )


def test_predict_chunks_universal_kriging_with_specified_drift(samples2d):
    samples2d['depth'] = samples2d.points[:, 0]
    model = samples2d.krige.fit(
        'elevation', method='universal', variogram_model='linear', specified_drift=['depth']
    )
    target = pv.ImageData(dimensions=(7, 6, 1), spacing=(1.5, 2.0, 1.0))
    target['depth'] = target.points[:, 0]
    whole = model.predict(target, max_memory=None)
    chunked = model.predict(target, max_memory=3 * 50 * 8)
    np.testing.assert_allclose(chunked['elevation'], whole['elevation'])


def test_max_distance_skips_points_far_from_every_sample(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    target = pv.PolyData([[5.0, 5.0, 0.0], [40.0, 40.0, 0.0], [10.5, 10.0, 0.0]])
    out = model.predict(target, max_distance=3.0)
    np.testing.assert_array_equal(out['vtkValidPointMask'], [1, 0, 1])
    assert np.isnan(out['elevation'][1])
    assert np.isnan(out['elevation variance'][1])
    full = model.predict(target)
    np.testing.assert_allclose(out['elevation'][[0, 2]], full['elevation'][[0, 2]])


def test_max_std_nulls_uncertain_points_and_combines_with_where(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    target = pv.ImageData(dimensions=(30, 30, 1), spacing=(1.0, 1.0, 1.0), origin=(-10, -10, 0))
    full = model.predict(target, std=True)
    limit = np.median(full['elevation std'])
    where = target.points[:, 1] > 0
    out = model.predict(target, std=True, max_std=limit, where=where)
    keep = where & (full['elevation std'] <= limit)
    np.testing.assert_array_equal(out['vtkValidPointMask'], keep.astype(np.uint8))
    np.testing.assert_allclose(out['elevation'][keep], full['elevation'][keep])
    assert np.isnan(out['elevation'][~keep]).all()


def test_to_image_forwards_extrapolation_masks(samples2d):
    image = samples2d.krige.to_image(
        'elevation', dimensions=(20, 20), padding=0.5, max_distance=1.5
    )
    mask = image['vtkValidPointMask'].astype(bool)
    assert 0 < mask.sum() < image.n_points


@pytest.mark.parametrize('fixture', ['samples2d', 'samples3d'])
def test_estimates_are_invariant_to_large_coordinate_offsets(request, fixture):
    samples = request.getfixturevalue(fixture)
    scalars = samples.array_names[0]
    target = pv.ImageData(dimensions=(5, 5, 2), spacing=(2.0, 2.0, 2.0), origin=(1.0, 1.0, 1.0))
    if fixture == 'samples2d':
        target = pv.ImageData(
            dimensions=(5, 5, 1), spacing=(2.0, 2.0, 1.0), origin=(1.0, 1.0, 0.0)
        )
    offset = np.array([512_345.0, 4_123_456.0, 0.0])
    local = samples.krige.fit(scalars, variogram_model='spherical').predict(target)
    utm = samples.translate(offset).krige.fit(scalars, variogram_model='spherical')
    shifted = utm.predict(target.translate(offset))
    np.testing.assert_allclose(shifted[scalars], local[scalars], rtol=1e-6, atol=1e-6)


def test_variance_output_can_be_renamed_skipped_or_written_as_std(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    target = pv.PolyData([[30.0, 30.0, 0.0], [2.0, 8.0, 0.0]])
    default = model.predict(target)
    renamed = model.predict(target, variance='sigma2')
    skipped = model.predict(target, variance=False)
    std = model.predict(target, std=True)
    np.testing.assert_allclose(renamed['sigma2'], default['elevation variance'])
    assert skipped.array_names == ['elevation']
    np.testing.assert_allclose(std['elevation std'] ** 2, default['elevation variance'])
    assert 'elevation variance' not in std.array_names


@pytest.mark.parametrize('as_image', [True, False])
def test_where_restricts_kriging_and_writes_valid_point_mask(samples2d, as_image):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    target = pv.ImageData(dimensions=(6, 4, 1), spacing=(2.0, 3.0, 1.0))
    if not as_image:
        target = target.cast_to_pointset()
    inside = target.points[:, 0] < 5.0
    target.point_data['inside'] = inside
    full = model.predict(target)
    out = model.predict(target, where='inside', null_value=-999.0)
    np.testing.assert_allclose(out['elevation'][inside], full['elevation'][inside])
    np.testing.assert_array_equal(out['elevation'][~inside], -999.0)
    np.testing.assert_array_equal(out['vtkValidPointMask'], inside.astype(np.uint8))


def test_to_image_spans_sample_bounds_and_matches_pykrige(samples2d):
    image = samples2d.krige.to_image('elevation', dimensions=(11, 6), variogram_model='linear')
    assert isinstance(image, pv.ImageData)
    assert image.dimensions == (11, 6, 1)
    np.testing.assert_allclose(image.bounds, samples2d.bounds)
    oracle = OrdinaryKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='linear',
    )
    expected, _ = oracle.execute('points', image.points[:, 0], image.points[:, 1])
    np.testing.assert_allclose(image['elevation'], np.ma.getdata(expected))


def test_to_image_3d_with_spacing_and_padding(samples3d):
    image = samples3d.krige.to_image('temperature', spacing=1.0, padding=0.1, n_closest_points=10)
    xmin, xmax, ymin, ymax, zmin, zmax = samples3d.bounds
    np.testing.assert_allclose(
        image.origin,
        (xmin - 0.1 * (xmax - xmin), ymin - 0.1 * (ymax - ymin), zmin - 0.1 * (zmax - zmin)),
    )
    assert image.spacing == (1.0, 1.0, 1.0)
    assert image.bounds[1] >= xmax
    assert image.bounds[5] >= zmax
    assert not np.isnan(image['temperature']).any()


def test_predict_leaves_target_untouched_and_chains_into_filters(samples2d):
    target = pv.Plane(center=(5, 5, 0), i_size=10, j_size=10)
    before = target.copy()
    out = samples2d.krige.fit('elevation').predict(target)
    assert out is not target
    assert target == before
    contours = out.contour(isosurfaces=3, scalars='elevation')
    assert contours.n_points > 0
