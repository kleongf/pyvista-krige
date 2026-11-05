from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige


def manual_folds(samples, scalars, folds, fit_kwargs, predict_kwargs):
    estimates = np.empty(samples.n_points)
    variances = np.empty(samples.n_points)
    for fold in np.unique(folds):
        test = folds == fold
        train = pv.PolyData(samples.points[~test])
        for name in samples.point_data:
            train[name] = samples[name][~test]
        target = pv.PolyData(samples.points[test])
        for name in samples.point_data:
            target[name] = samples[name][test]
        out = train.krige.fit(scalars, **fit_kwargs).predict(target, **predict_kwargs)
        estimates[test] = out[scalars]
        variances[test] = out[f'{scalars} variance']
    return estimates, variances


@pytest.mark.parametrize(
    ('fixture', 'fit_kwargs', 'predict_kwargs'),
    [
        ('samples2d', {'variogram_model': 'spherical'}, {}),
        ('samples3d', {'variogram_model': 'linear'}, {'n_closest_points': 10}),
        (
            'samples3d',
            {'method': 'universal', 'drift_terms': ['regional_linear'], 'detrend': True},
            {},
        ),
    ],
)
def test_cross_validate_equals_manual_refits(request, fixture, fit_kwargs, predict_kwargs):
    samples = request.getfixturevalue(fixture)
    scalars = samples.array_names[0]
    model = samples.krige.fit(scalars, **fit_kwargs)
    holdout = model.cross_validate(folds=4, seed=7, **predict_kwargs)
    assert isinstance(holdout, pyvista_krige.HoldOut)
    folds = np.random.default_rng(7).permutation(samples.n_points) % 4
    estimate, variance = manual_folds(samples, scalars, folds, fit_kwargs, predict_kwargs)
    error = estimate - samples[scalars]
    np.testing.assert_allclose(holdout.samples[f'{scalars} error'], error)
    np.testing.assert_allclose(
        holdout.samples[f'{scalars} z-score'], error / np.sqrt(variance), rtol=1e-6
    )
    assert holdout.rmse == pytest.approx(np.sqrt(np.mean(error**2)))
    assert holdout.mae == pytest.approx(np.mean(np.abs(error)))
    assert holdout.bias == pytest.approx(np.mean(error))


def test_leave_one_out_uses_one_fold_per_sample(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    holdout = model.cross_validate(folds='loo')
    folds = np.arange(samples2d.n_points)
    estimate, _ = manual_folds(samples2d, 'elevation', folds, {'variogram_model': 'spherical'}, {})
    np.testing.assert_allclose(
        holdout.samples['elevation error'], estimate - samples2d['elevation']
    )


def test_cross_validate_carries_specified_drift_arrays(samples2d):
    samples2d['depth'] = samples2d.points[:, 0]
    model = samples2d.krige.fit(
        'elevation', method='universal', variogram_model='linear', specified_drift=['depth']
    )
    holdout = model.cross_validate(folds=3)
    assert np.isfinite(holdout.rmse)
    assert 'depth' in holdout.samples.point_data


def test_cross_validate_rejects_bad_fold_counts(samples2d):
    model = samples2d.krige.fit('elevation')
    with pytest.raises(ValueError, match='folds must be'):
        model.cross_validate(folds=1)
