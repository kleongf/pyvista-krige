from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import pyvista as pv

from pyvista_krige import KrigingModel
from pyvista_krige import NormalScoreTransform


def round_trip(model, tmp_path):
    path = tmp_path / 'model.npz'
    model.save(path)
    return KrigingModel.load(path)


def assert_predicts_the_same(model, loaded, target):
    before, after = model.predict(target), loaded.predict(target)
    assert set(after.point_data) == set(before.point_data)
    for name in before.point_data:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def plane_grid(samples):
    lower, upper = samples.points.min(axis=0), samples.points.max(axis=0)
    return pv.ImageData(
        dimensions=(9, 7, 1), origin=lower, spacing=((upper - lower) / [8, 6, 1]) * [1, 1, 0]
    )


def box_grid(samples):
    lower, upper = samples.points.min(axis=0), samples.points.max(axis=0)
    return pv.ImageData(dimensions=(6, 5, 4), origin=lower, spacing=(upper - lower) / [5, 4, 3])


def test_ordinary_2d_round_trip_keeps_the_fitted_variogram(samples2d, tmp_path):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    loaded = round_trip(model, tmp_path)
    assert_predicts_the_same(model, loaded, plane_grid(samples2d))
    assert loaded.variogram.model_name == 'spherical'
    np.testing.assert_array_equal(loaded.variogram.parameters, model.variogram.parameters)
    assert loaded.dim == 2
    assert loaded.n_samples == model.n_samples
    np.testing.assert_array_equal(loaded.samples['elevation'], model.samples['elevation'])


def test_universal_with_specified_drift(samples2d, tmp_path):
    samples2d['depth'] = samples2d.points[:, 0] ** 2
    model = samples2d.krige.fit(
        'elevation', method='universal', variogram_model='linear', specified_drift=['depth']
    )
    loaded = round_trip(model, tmp_path)
    target = plane_grid(samples2d)
    target['depth'] = target.points[:, 0] ** 2
    assert_predicts_the_same(model, loaded, target)
    np.testing.assert_array_equal(loaded.samples['depth'], model.samples['depth'])


def test_universal_with_external_drift_image(samples2d, tmp_path):
    dem = pv.ImageData(dimensions=(12, 12, 1), origin=(-0.5, -0.5, 0.0))
    dem['height'] = np.hypot(dem.points[:, 0] - 5, dem.points[:, 1] - 5)
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='linear',
        external_drift=dem,
        external_drift_scalars='height',
    )
    loaded = round_trip(model, tmp_path)
    assert_predicts_the_same(model, loaded, plane_grid(samples2d))


def test_detrended_variogram_is_not_refitted(samples2d, tmp_path):
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='exponential',
        drift_terms=['regional_linear'],
        detrend=True,
        max_lag=6.0,
    )
    loaded = round_trip(model, tmp_path)
    assert_predicts_the_same(model, loaded, plane_grid(samples2d))
    before, after = model.variogram, loaded.variogram
    np.testing.assert_array_equal(after.parameters, before.parameters)
    np.testing.assert_array_equal(after.lags, before.lags)
    np.testing.assert_array_equal(after.semivariance, before.semivariance)
    np.testing.assert_array_equal(after.counts, before.counts)


def test_ordinary_3d_with_anisotropy(samples3d, tmp_path):
    model = samples3d.krige.fit(
        'temperature',
        variogram_model='spherical',
        anisotropy_scaling_z=2.0,
        anisotropy_angle_z=30.0,
    )
    loaded = round_trip(model, tmp_path)
    assert loaded.dim == 3
    assert loaded.anisotropy == model.anisotropy
    assert_predicts_the_same(model, loaded, box_grid(samples3d))


def test_universal_3d_with_drift(samples3d, tmp_path):
    model = samples3d.krige.fit(
        'temperature',
        method='universal',
        variogram_model='linear',
        drift_terms=['regional_linear'],
    )
    loaded = round_trip(model, tmp_path)
    assert_predicts_the_same(model, loaded, box_grid(samples3d))


def test_estimated_anisotropy_is_kept_not_estimated_again(tmp_path):
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 20, (60, 3)) * [1, 1, 0]
    samples = pv.PolyData(points)
    samples['value'] = np.sin(points[:, 0] / 2) + 0.1 * points[:, 1]
    model = samples.krige.fit('value', variogram_model='exponential', anisotropy='auto')
    loaded = round_trip(model, tmp_path)
    assert loaded.anisotropy == model.anisotropy
    assert_predicts_the_same(model, loaded, plane_grid(samples))


@pytest.fixture
def skewed() -> pv.PolyData:
    rng = np.random.default_rng(3)
    samples = pv.PolyData(rng.uniform(0, 10, (40, 3)) * [1, 1, 0])
    samples['grade'] = np.exp(np.sin(samples.points[:, 0] / 2) + 0.5 * rng.standard_normal(40))
    return samples


def test_log_transform(skewed, tmp_path):
    model = skewed.krige.fit('grade', variogram_model='exponential', transform='log')
    loaded = round_trip(model, tmp_path)
    assert loaded.transform.name == 'log'
    assert_predicts_the_same(model, loaded, plane_grid(skewed))


def test_normal_score_transform_keeps_its_table_and_limits(skewed, tmp_path):
    model = skewed.krige.fit(
        'grade', variogram_model='exponential', transform='normal_score', transform_limits=(0, 50)
    )
    loaded = round_trip(model, tmp_path)
    assert isinstance(loaded.transform, NormalScoreTransform)
    np.testing.assert_array_equal(loaded.transform.values, model.transform.values)
    np.testing.assert_array_equal(loaded.transform.cdf, model.transform.cdf)
    assert loaded.transform.values[0] == 0
    assert loaded.transform.values[-1] == 50
    assert_predicts_the_same(model, loaded, plane_grid(skewed))


def test_loaded_model_refits_like_the_saved_one(skewed, tmp_path):
    model = skewed.krige.fit(
        'grade', variogram_model='exponential', transform='normal_score', transform_limits=(0, 50)
    )
    loaded = round_trip(model, tmp_path)
    before, after = model.cross_validate(4), loaded.cross_validate(4)
    assert after.rmse == before.rmse
    np.testing.assert_array_equal(after.samples['grade error'], before.samples['grade error'])


def test_path_is_written_as_given(samples2d, tmp_path):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    path = tmp_path / 'model.krige'
    model.save(str(path))
    assert path.exists()
    assert_predicts_the_same(model, KrigingModel.load(str(path)), plane_grid(samples2d))


def rewrite_metadata(path, **changes):
    """Rewrite the JSON metadata of a saved model with ``changes``."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays['metadata']))
    metadata.update(changes)
    arrays['metadata'] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


def test_newer_format_version_is_rejected(samples2d, tmp_path):
    path = tmp_path / 'model.npz'
    samples2d.krige.fit('elevation').save(path)
    rewrite_metadata(path, version=2)
    with pytest.raises(ValueError, match='format version 2'):
        KrigingModel.load(path)


def test_other_npz_files_are_rejected(tmp_path):
    path = tmp_path / 'other.npz'
    np.savez(path, metadata=np.array('{"format": "something else"}'))
    with pytest.raises(ValueError, match='not a pyvista-krige model'):
        KrigingModel.load(path)
    np.savez(path, values=np.arange(3))
    with pytest.raises(ValueError, match='not a pyvista-krige model'):
        KrigingModel.load(path)


def test_file_holds_no_pickled_objects(samples2d, tmp_path):
    path = tmp_path / 'model.npz'
    samples2d.krige.fit('elevation', variogram_model='spherical').save(path)
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)
    with zipfile.ZipFile(path) as archive:
        assert all(name.endswith('.npy') for name in archive.namelist())


def test_custom_variogram_cannot_be_saved(samples2d, tmp_path):
    model = samples2d.krige.fit(
        'elevation',
        variogram_model='custom',
        variogram_parameters=[1.0, 0.0],
        variogram_function=lambda p, d: p[0] * d + p[1],
    )
    with pytest.raises(TypeError, match='custom function'):
        model.save(tmp_path / 'model.npz')


def test_functional_drift_cannot_be_saved(samples2d, tmp_path):
    model = samples2d.krige.fit(
        'elevation',
        method='universal',
        variogram_model='linear',
        functional_drift=[lambda x, y: x * y],
    )
    with pytest.raises(TypeError, match='functional_drift'):
        model.save(tmp_path / 'model.npz')
    assert not (tmp_path / 'model.npz').exists()


def test_numpy_options_round_trip(samples2d, tmp_path):
    model = samples2d.krige.fit(
        'elevation',
        variogram_model='spherical',
        variogram_parameters=[np.float64(1.0), 6.0, 0.0],
        nlags=np.int64(8),
        weight=np.True_,
    )
    loaded = round_trip(model, tmp_path)
    assert_predicts_the_same(model, loaded, plane_grid(samples2d))
    np.testing.assert_array_equal(loaded.variogram.lags, model.variogram.lags)
