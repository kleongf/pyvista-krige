from __future__ import annotations

import sys

import numpy as np
import pytest
import pyvista as pv
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging


def depth(points):
    return 0.5 * points[:, 0] - 0.2 * points[:, 1]


@pytest.mark.parametrize('as_image', [True, False])
def test_specified_drift_is_read_by_name_from_samples_and_target(samples2d, as_image):
    samples2d['depth'] = depth(samples2d.points)
    target = pv.ImageData(dimensions=(5, 4, 1), spacing=(2.0, 3.0, 1.0))
    if not as_image:
        target = target.cast_to_pointset()
    target['depth'] = depth(target.points)
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='linear',
        drift_terms=['specified'],
        specified_drift=['depth'],
    )
    out = model.predict(target)
    oracle = UniversalKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='linear',
        drift_terms=['specified'],
        specified_drift=[samples2d['depth']],
    )
    expected, _ = oracle.execute(
        'points',
        target.points[:, 0],
        target.points[:, 1],
        specified_drift_arrays=[target['depth']],
    )
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_external_drift_accepts_a_2d_image(samples2d):
    dem = pv.ImageData(dimensions=(12, 12, 1), spacing=(1.0, 1.0, 1.0), origin=(-0.5, -0.5, 0.0))
    dem['height'] = np.hypot(dem.points[:, 0] - 5, dem.points[:, 1] - 5)
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='linear',
        drift_terms=['external_Z'],
        external_drift=dem,
        external_drift_scalars='height',
    )
    target = pv.PolyData([[2.0, 3.0, 0.0], [7.5, 6.0, 0.0]])
    out = model.predict(target)
    x = -0.5 + np.arange(12.0)
    oracle = UniversalKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='linear',
        drift_terms=['external_Z'],
        external_drift=dem['height'].reshape(12, 12),
        external_drift_x=x,
        external_drift_y=x,
    )
    expected, _ = oracle.execute('points', target.points[:, 0], target.points[:, 1])
    np.testing.assert_allclose(out['elevation'], np.ma.getdata(expected))


def test_external_drift_image_must_be_flat_and_axis_aligned(samples2d):
    volume = pv.ImageData(dimensions=(4, 4, 3))
    volume['height'] = np.zeros(volume.n_points)
    with pytest.raises(ValueError, match='single-layer, axis-aligned ImageData'):
        samples2d.krige.fit(
            'elevation',
            method='universal',
            drift_terms=['external_Z'],
            external_drift=volume,
            external_drift_scalars='height',
        )


def test_gstools_model_matches_pykrige_and_sets_dimension(samples2d, samples3d):
    gs = pytest.importorskip('gstools')
    cov = gs.Spherical(dim=2, var=1.0, len_scale=4.0)
    model = samples2d.krige.fit('elevation', variogram_model=cov)
    target = pv.PolyData([[3.0, 3.0, 0.0], [8.0, 1.0, 0.0]])
    oracle = OrdinaryKriging(
        samples2d.points[:, 0], samples2d.points[:, 1], samples2d['elevation'], variogram_model=cov
    )
    expected, _ = oracle.execute('points', target.points[:, 0], target.points[:, 1])
    np.testing.assert_allclose(model.predict(target)['elevation'], np.ma.getdata(expected))
    assert model.variogram.model_name == 'custom'
    assert samples3d.krige.fit('temperature', variogram_model=cov).dim == 2


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'variogram_model': 'gs3', 'dim': 2}, 'GSTools model has dim=3 but dim=2'),
        ({'coordinates_type': 'geographic', 'method': 'universal'}, 'only with 2D ordinary'),
        ({'coordinates_type': 'geographic', 'dim': 3}, 'only with 2D ordinary'),
    ],
)
def test_incompatible_model_options_are_rejected(samples3d, kwargs, match):
    if kwargs.get('variogram_model') == 'gs3':
        kwargs['variogram_model'] = pytest.importorskip('gstools').Spherical(dim=3)
    with pytest.raises(ValueError, match=match):
        samples3d.krige.fit('temperature', **kwargs)


def with_predictors(dataset):
    dataset['slope'] = 0.3 * dataset.points[:, 0]
    dataset['aspect'] = np.cos(dataset.points[:, 1])
    return dataset


def test_regression_kriging_matches_pykrige(samples2d):
    linear_model = pytest.importorskip('sklearn.linear_model')
    from pykrige.rk import RegressionKriging

    samples = with_predictors(samples2d)
    target = with_predictors(pv.PolyData([[1.0, 1.0, 0.0], [6.0, 4.0, 0.0], [9.0, 9.0, 0.0]]))
    out = samples.krige.regression(
        target,
        'elevation',
        predictors=['slope', 'aspect'],
        regression_model=linear_model.LinearRegression(),
        variogram_model='linear',
    )
    oracle = RegressionKriging(
        regression_model=linear_model.LinearRegression(),
        method='ordinary',
        variogram_model='linear',
    )
    p_samples = np.column_stack([samples['slope'], samples['aspect']])
    oracle.fit(p_samples, samples.points[:, :2], samples['elevation'])
    p_target = np.column_stack([target['slope'], target['aspect']])
    expected = oracle.predict(p_target, target.points[:, :2])
    np.testing.assert_allclose(out['elevation'], expected)


def test_classification_kriging_matches_pykrige(samples2d):
    linear_model = pytest.importorskip('sklearn.linear_model')
    from pykrige.ck import ClassificationKriging

    samples = with_predictors(samples2d)
    samples['rock'] = (samples['elevation'] > np.median(samples['elevation'])).astype(int)
    target = with_predictors(pv.PolyData([[1.0, 1.0, 0.0], [6.0, 4.0, 0.0], [9.0, 9.0, 0.0]]))
    out = samples.krige.classify(
        target,
        'rock',
        predictors=['slope', 'aspect'],
        classification_model=linear_model.LogisticRegression(),
        variogram_model='linear',
    )
    oracle = ClassificationKriging(
        classification_model=linear_model.LogisticRegression(),
        method='ordinary',
        variogram_model='linear',
    )
    oracle.fit(
        np.column_stack([samples['slope'], samples['aspect']]),
        samples.points[:, :2],
        samples['rock'].reshape(-1, 1),
    )
    expected = oracle.predict(
        np.column_stack([target['slope'], target['aspect']]), target.points[:, :2]
    )
    np.testing.assert_array_equal(out['rock'], expected)
    assert np.issubdtype(out['rock'].dtype, np.integer)


def test_cross_validate_picks_best_variogram_like_grid_search(samples2d):
    model_selection = pytest.importorskip('sklearn.model_selection')
    from pykrige.rk import Krige

    grid = {'variogram_model': ['linear', 'gaussian', 'spherical']}
    result = samples2d.krige.cross_validate('elevation', grid, cv=4)
    oracle = model_selection.GridSearchCV(Krige(method='ordinary'), grid, cv=4)
    oracle.fit(samples2d.points[:, :2], samples2d['elevation'])
    assert result.best_params == oracle.best_params_
    assert result.best_score == pytest.approx(oracle.best_score_)
    np.testing.assert_allclose(
        result.results['mean_test_score'], oracle.cv_results_['mean_test_score']
    )
    assert result.model.variogram.model_name == oracle.best_params_['variogram_model']


def test_sklearn_methods_explain_how_to_install_the_extra(samples2d, monkeypatch):
    monkeypatch.setitem(sys.modules, 'sklearn', None)  # makes `import sklearn` fail
    samples = with_predictors(samples2d)
    with pytest.raises(ImportError, match=r'pyvista-krige\[sklearn\]'):
        samples.krige.regression(samples, 'elevation', predictors=['slope'])
    with pytest.raises(ImportError, match=r'pyvista-krige\[sklearn\]'):
        samples.krige.cross_validate('elevation', {'variogram_model': ['linear']})
