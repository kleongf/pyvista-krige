from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv


@pytest.fixture
def grid2d() -> pv.ImageData:
    return pv.ImageData(dimensions=(21, 21, 1), spacing=(0.5, 0.5, 1.0), origin=(0, 0, 0))


@pytest.fixture
def grid3d() -> pv.ImageData:
    return pv.ImageData(dimensions=(11, 11, 6), spacing=(1.0, 1.0, 1.0))


def spherical(samples, scalars, **kwargs):
    return samples.krige.fit(scalars, variogram_model='spherical', **kwargs)


def test_first_pick_is_the_point_of_maximum_std(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    std = model.predict(grid2d, std=True)['elevation std']
    picks = model.suggest_samples(grid2d, n=3)
    assert isinstance(picks, pv.PolyData)
    assert picks.n_points == 3
    np.testing.assert_allclose(picks.points[0], grid2d.points[np.argmax(std)])
    assert picks['std before pick'][0] == pytest.approx(std.max())


def test_picks_carry_order_and_std_arrays(samples2d, grid2d):
    picks = spherical(samples2d, 'elevation').suggest_samples(grid2d, n=4)
    np.testing.assert_array_equal(picks['pick order'], [1, 2, 3, 4])
    assert np.all(np.diff(picks['std before pick']) <= 1e-9)
    assert np.all(picks['std before pick'] > 0)


def test_picks_are_distinct(samples2d, grid2d):
    picks = spherical(samples2d, 'elevation').suggest_samples(grid2d, n=15)
    assert len(np.unique(picks.points, axis=0)) == 15


def test_std_is_about_zero_at_the_picks_once_they_are_samples(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    picks = model.suggest_samples(grid2d, n=4)
    parameters = model.variogram.parameters
    names = ('psill', 'range', 'nugget')
    both = pv.PolyData(np.vstack([samples2d.points, picks.points]))
    both['elevation'] = np.zeros(both.n_points)
    augmented = both.krige.fit(
        'elevation',
        variogram_model='spherical',
        variogram_parameters=dict(zip(names, parameters, strict=True)),
    )
    std = augmented.predict(picks, std=True)['elevation std']
    assert std.max() < 1e-4
    # The pick that was made last reduced the std there from its recorded value to about zero.
    assert picks['std before pick'].min() > 100 * std.max()


def test_the_model_and_target_are_left_unchanged(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    before = model.variogram.parameters.copy()
    arrays = list(grid2d.point_data)
    model.suggest_samples(grid2d, n=3)
    assert model.n_samples == samples2d.n_points
    assert list(grid2d.point_data) == arrays
    np.testing.assert_array_equal(model.variogram.parameters, before)


def test_min_spacing_keeps_picks_apart(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    free = model.suggest_samples(grid2d, n=6)
    spacing = 3.0
    spaced = model.suggest_samples(grid2d, n=6, min_spacing=spacing)
    gaps = np.linalg.norm(spaced.points[:, None] - spaced.points[None], axis=2)
    gaps = gaps[np.triu_indices(6, k=1)]
    assert gaps.min() >= spacing
    nearest = np.linalg.norm(free.points[:, None] - free.points[None], axis=2)
    assert nearest[np.triu_indices(6, k=1)].min() < spacing


def test_where_restricts_the_candidates(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    allowed = grid2d.points[:, 0] > 7.0
    grid2d['allowed'] = allowed
    by_name = model.suggest_samples(grid2d, n=4, where='allowed')
    by_mask = model.suggest_samples(grid2d, n=4, where=allowed)
    assert np.all(by_name.points[:, 0] > 7.0)
    np.testing.assert_array_equal(by_name.points, by_mask.points)
    std = model.predict(grid2d, std=True)['elevation std']
    std[~allowed] = -1
    np.testing.assert_allclose(by_name.points[0], grid2d.points[np.argmax(std)])


def test_universal_kriging_reads_drift_values_from_the_target(samples2d, grid2d):
    samples2d['height'] = samples2d.points[:, 0] * 0.3 + samples2d.points[:, 1] ** 2 * 0.01
    grid2d['height'] = grid2d.points[:, 0] * 0.3 + grid2d.points[:, 1] ** 2 * 0.01
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='spherical',
        specified_drift=['height'],
    )
    std = model.predict(grid2d, std=True)['elevation std']
    picks = model.suggest_samples(grid2d, n=4)
    np.testing.assert_allclose(picks.points[0], grid2d.points[np.argmax(std)])
    assert picks['std before pick'][0] == pytest.approx(std.max())
    assert len(np.unique(picks.points, axis=0)) == 4
    assert np.all(np.diff(picks['std before pick']) <= 1e-9)


def test_universal_kriging_needs_the_drift_array_on_the_target(samples2d, grid2d):
    samples2d['height'] = samples2d.points[:, 0]
    model = samples2d.krige.fit(
        'elevation', method='universal', variogram_model='linear', specified_drift=['height']
    )
    with pytest.raises(KeyError, match='height'):
        model.suggest_samples(grid2d, n=2)


def test_regional_linear_universal_kriging(samples2d, grid2d):
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='spherical',
        drift_terms=['regional_linear'],
    )
    std = model.predict(grid2d, std=True)['elevation std']
    picks = model.suggest_samples(grid2d, n=3)
    np.testing.assert_allclose(picks.points[0], grid2d.points[np.argmax(std)])


def test_three_dimensional_model(samples3d, grid3d):
    model = spherical(samples3d, 'temperature')
    std = model.predict(grid3d, std=True)['temperature std']
    picks = model.suggest_samples(grid3d, n=3, min_spacing=2.0)
    np.testing.assert_allclose(picks.points[0], grid3d.points[np.argmax(std)])
    assert len(np.unique(picks.points, axis=0)) == 3
    assert np.all(np.diff(picks['std before pick']) <= 1e-9)


def test_point_cloud_target(samples2d):
    rng = np.random.default_rng(3)
    cloud = pv.PolyData(np.column_stack([rng.uniform(0, 10, (60, 2)), np.zeros(60)]))
    model = spherical(samples2d, 'elevation')
    std = model.predict(cloud, std=True)['elevation std']
    picks = model.suggest_samples(cloud, n=5)
    np.testing.assert_allclose(picks.points[0], cloud.points[np.argmax(std)])


def test_gstools_model(samples2d, grid2d):
    gs = pytest.importorskip('gstools')
    model = samples2d.krige.fit(
        'elevation', variogram_model=gs.Spherical(dim=2, var=1, len_scale=4)
    )
    std = model.predict(grid2d, std=True)['elevation std']
    picks = model.suggest_samples(grid2d, n=3)
    np.testing.assert_allclose(picks.points[0], grid2d.points[np.argmax(std)])
    assert len(np.unique(picks.points, axis=0)) == 3


def test_fewer_picks_than_asked_warns_when_spacing_runs_out(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    with pytest.warns(UserWarning, match='Only 1 of 5'):
        picks = model.suggest_samples(grid2d, n=5, min_spacing=100.0)
    assert picks.n_points == 1


def test_invalid_arguments(samples2d, grid2d):
    model = spherical(samples2d, 'elevation')
    with pytest.raises(ValueError, match='n must be'):
        model.suggest_samples(grid2d, n=0)
    with pytest.raises(ValueError, match='candidate'):
        model.suggest_samples(grid2d, n=grid2d.n_points + 1)
    with pytest.raises(ValueError, match='min_spacing'):
        model.suggest_samples(grid2d, n=2, min_spacing=-1.0)
    with pytest.raises(TypeError, match='MultiBlock'):
        model.suggest_samples(pv.MultiBlock([grid2d]), n=2)
