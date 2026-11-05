from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
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


def trended_field(n=300, seed=3):
    """Points in a box with a strong linear z trend plus a short-range random field."""
    rng = np.random.default_rng(seed)
    points = rng.uniform((0, 0, 0), (20, 20, 10), size=(n, 3))
    centers = rng.uniform((0, 0, 0), (20, 20, 10), size=(40, 3))
    bumps = rng.normal(size=40)
    dist = np.linalg.norm(points[:, None] - centers[None], axis=-1)
    noise = (bumps * np.exp(-((dist / 2.0) ** 2))).sum(axis=1)
    samples = pv.PolyData(points)
    samples['t'] = 10.0 * points[:, 2] + noise
    samples['noise'] = noise
    return samples


def test_detrend_fits_variogram_to_drift_residuals():
    samples = trended_field()
    kwargs = {'variogram_model': 'exponential', 'nlags': 10}
    raw = samples.krige.fit('t', method='universal', drift_terms=['regional_linear'], **kwargs)
    detrended = samples.krige.fit(
        't', method='universal', drift_terms=['regional_linear'], detrend=True, **kwargs
    )
    noise_only = samples.krige.fit('noise', **kwargs)
    sill = detrended.variogram.parameters[0] + detrended.variogram.parameters[2]
    noise_sill = noise_only.variogram.parameters[0] + noise_only.variogram.parameters[2]
    raw_sill = raw.variogram.parameters[0] + raw.variogram.parameters[2]
    assert raw_sill > 20 * noise_sill  # the trend dominates PyKrige's own fit
    assert sill == pytest.approx(noise_sill, rel=0.5)
    # The kriging system uses the residual variogram, not just the reported one.
    np.testing.assert_allclose(
        detrended.instance.variogram_model_parameters, detrended.variogram.parameters
    )
    assert detrended.variogram.semivariance.max() < 0.1 * raw.variogram.semivariance.max()


def test_detrend_supports_specified_and_functional_drift():
    samples = trended_field()
    samples['depth'] = samples.points[:, 2]
    specified = samples.krige.fit(
        't', method='universal', specified_drift=['depth'], detrend=True, nlags=10
    )
    functional = samples.krige.fit(
        't', method='universal', functional_drift=[lambda x, y, z: z], detrend=True, nlags=10
    )
    np.testing.assert_allclose(
        specified.variogram.semivariance, functional.variogram.semivariance, rtol=1e-6
    )


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'detrend': True}, 'detrend needs universal kriging'),
        ({'method': 'universal', 'detrend': True}, 'detrend needs at least one drift term'),
    ],
)
def test_detrend_rejects_models_without_drift(samples3d, kwargs, match):
    with pytest.raises(ValueError, match=match):
        samples3d.krige.fit('temperature', **kwargs)


def test_max_lag_trims_long_lags_from_the_fit(samples3d):
    full = samples3d.krige.fit('temperature', variogram_model='spherical', nlags=8)
    trimmed = samples3d.krige.fit('temperature', variogram_model='spherical', nlags=8, max_lag=4.0)
    assert full.variogram.lags.max() > 4.0
    assert trimmed.variogram.lags.max() <= 4.0
    assert len(trimmed.variogram.lags) == 8
    assert trimmed.variogram.counts.sum() > 0
    np.testing.assert_allclose(
        trimmed.instance.variogram_model_parameters, trimmed.variogram.parameters
    )


def test_experimental_variogram_matches_brute_force_pairs(samples3d):
    vario = samples3d.krige.variogram('temperature', nlags=5, max_lag=6.0)
    points, values = samples3d.points, samples3d['temperature']
    i, j = np.triu_indices(len(values), k=1)
    dist = np.linalg.norm(points[i] - points[j], axis=1)
    gamma = 0.5 * (values[i] - values[j]) ** 2
    edges = np.linspace(0.0, 6.0, 6)
    for k in range(5):
        inside = (dist > edges[k]) & (dist <= edges[k + 1])
        assert vario.counts[k] == inside.sum()
        assert vario.semivariance[k] == pytest.approx(gamma[inside].mean())
        assert vario.lags[k] == pytest.approx(dist[inside].mean())


def test_directional_variograms_separate_vertical_and_horizontal_structure():
    rng = np.random.default_rng(5)
    points = rng.uniform((0, 0, 0), (20, 20, 20), size=(400, 3))
    samples = pv.PolyData(points)
    samples['layered'] = np.sin(points[:, 2] * 1.5)  # varies with z only
    # Stay under half the period (2.09) so the vertical variogram keeps rising.
    vertical = samples.krige.variogram('layered', direction='vertical', max_lag=2.0, nlags=2)
    horizontal = samples.krige.variogram('layered', direction='horizontal', max_lag=2.0, nlags=2)
    assert vertical.direction == 'vertical'
    assert np.all(vertical.semivariance > 4 * horizontal.semivariance)
    x_axis = samples.krige.variogram(
        'layered', direction=(1, 0, 0), tolerance=5.0, max_lag=2.0, nlags=2
    )
    np.testing.assert_allclose(x_axis.semivariance, 0.0, atol=0.02)


def test_experimental_variogram_fit_returns_model(samples3d):
    # The field has a linear trend, so its semivariance grows like a power law.
    vario = samples3d.krige.variogram('temperature', nlags=8, max_lag=6.0)
    fitted = vario.fit('power')
    assert fitted.model_name == 'power'
    assert fitted.parameters.shape == (3,)
    residual = fitted.model(vario.lags) - vario.semivariance
    assert np.sqrt(np.mean(residual**2)) < 0.3 * vario.semivariance.mean()


def test_with_variogram_keeps_anisotropy_and_lags(samples3d):
    model = samples3d.krige.fit(
        'temperature', variogram_model='linear', anisotropy_scaling_z=2.0, nlags=9
    )
    refit = model.with_variogram('spherical')
    assert refit.instance.anisotropy_scaling_z == 2.0
    assert len(refit.variogram.lags) == 9
