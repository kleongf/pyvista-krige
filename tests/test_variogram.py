from __future__ import annotations

import numpy as np
import pytest
from pykrige.ok import OrdinaryKriging

SPHERICAL = {'sill': 2.0, 'range': 4.0, 'nugget': 0.5}
# psill = 1.5: gamma(d) = 1.5 * (1.5 d/4 - 0.5 (d/4)^3) + 0.5 for d <= 4, else 2.0
SPHERICAL_AT = {0.0: 0.5, 1.0: 1.05078125, 2.0: 1.53125, 4.0: 2.0, 6.0: 2.0}


def test_variogram_exposes_experimental_lags_and_fitted_parameters(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical', nlags=8)
    oracle = OrdinaryKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='spherical',
        nlags=8,
    )
    vario = model.variogram
    assert vario.model_name == 'spherical'
    np.testing.assert_allclose(vario.lags, oracle.lags)
    np.testing.assert_allclose(vario.semivariance, oracle.semivariance)
    np.testing.assert_allclose(vario.parameters, oracle.variogram_model_parameters)


@pytest.mark.parametrize('fixture', ['samples2d', 'samples3d'])
def test_variogram_model_curve_follows_spherical_formula(request, fixture):
    samples = request.getfixturevalue(fixture)
    model = samples.krige.fit(
        samples.array_names[0], variogram_model='spherical', variogram_parameters=SPHERICAL
    )
    distances = np.array(list(SPHERICAL_AT))
    np.testing.assert_allclose(model.variogram.model(distances), list(SPHERICAL_AT.values()))
    np.testing.assert_allclose(
        model.variogram.model_values, model.variogram.model(model.variogram.lags)
    )


def test_statistics_match_pykrige_when_enabled(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='linear', enable_statistics=True)
    oracle = OrdinaryKriging(
        samples2d.points[:, 0],
        samples2d.points[:, 1],
        samples2d['elevation'],
        variogram_model='linear',
        enable_statistics=True,
    )
    stats = model.statistics
    np.testing.assert_allclose((stats.q1, stats.q2, stats.cr), oracle.get_statistics())
    np.testing.assert_allclose(stats.residuals, oracle.get_epsilon_residuals())


def test_statistics_explain_how_to_enable_them_for_2d_ordinary(samples2d):
    model = samples2d.krige.fit('elevation')
    with pytest.raises(ValueError, match='enable_statistics=True'):
        _ = model.statistics


def test_with_variogram_returns_refitted_model_and_keeps_original(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='linear')
    refit = model.with_variogram('spherical', variogram_parameters=SPHERICAL)
    assert refit is not model
    assert model.variogram.model_name == 'linear'
    assert refit.variogram.model_name == 'spherical'
    np.testing.assert_allclose(refit.variogram.model(np.array([2.0])), [1.53125])
