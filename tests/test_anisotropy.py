from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from pyvista_krige._anisotropy import estimate_anisotropy


def random_field(
    n: int,
    seed: int,
    *,
    covariance: str = 'gaussian',
    angle: float = 0.0,
    scales: tuple[float, ...] = (8.0, 8.0),
    extent: tuple[float, ...] = (100.0, 100.0),
    nugget: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random field with an anisotropic covariance at random points.

    ``scales`` are the correlation lengths along the field's own axes, which are
    rotated by ``angle`` degrees counter-clockwise from +x in the xy plane. The
    covariance is ``exp(-h**2)`` or ``exp(-h)`` in the scaled distance ``h``, plus a nugget.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(size=(n, len(extent))) * extent
    delta = points[:, None, :] - points[None, :, :]
    theta = np.radians(angle)
    along = delta[..., 0] * np.cos(theta) + delta[..., 1] * np.sin(theta)
    across = -delta[..., 0] * np.sin(theta) + delta[..., 1] * np.cos(theta)
    scaled = [along / scales[0], across / scales[1]]
    scaled.extend(delta[..., axis] / scales[axis] for axis in range(2, len(extent)))
    squared = sum(c**2 for c in scaled)
    kernel = np.exp(-squared) if covariance == 'gaussian' else np.exp(-np.sqrt(squared))
    matrix = kernel + (nugget + 1e-8) * np.eye(n)
    return points, np.linalg.cholesky(matrix) @ rng.standard_normal(n)


def as_samples(points: np.ndarray, values: np.ndarray) -> pv.PolyData:
    """Return the points (padded to 3D) with the values in a ``'value'`` array."""
    samples = pv.PolyData(np.column_stack([points, np.zeros(len(points))])[:, :3])
    samples['value'] = values
    return samples


def angle_error(a: float, b: float) -> float:
    """Return the difference of two orientations in degrees, modulo 180."""
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def test_estimate_recovers_angle_and_ratio_of_a_2d_field():
    # Correlation lengths 12 and 4 along 30 degrees, so the true ratio is 3.
    fits = [
        estimate_anisotropy(
            *random_field(350, seed, angle=30.0, scales=(12.0, 4.0)),
            'gaussian',
            nlags=12,
        )
        for seed in range(5)
    ]
    scaling = np.array([fit['anisotropy_scaling'] for fit in fits])
    assert set(fits[0]) == {'anisotropy_scaling', 'anisotropy_angle'}
    assert all(angle_error(fit['anisotropy_angle'], 30.0) < 20.0 for fit in fits)
    # One field gives a ratio from about 1.5 to 5 for a true ratio of 3.
    assert np.all((scaling > 1.4) & (scaling < 5.0))
    assert 1.8 < np.median(scaling) < 4.5


def test_estimate_recovers_a_layered_3d_field():
    # Horizontal correlation length 15, vertical 5. One field gives a vertical ratio anywhere
    # from about 1.5 to 5, so judge the median of several.
    fits = [
        estimate_anisotropy(
            *random_field(400, seed, scales=(15.0, 15.0, 5.0), extent=(80.0, 80.0, 50.0)),
            'gaussian',
            nlags=12,
        )
        for seed in range(5)
    ]
    assert set(fits[0]) == {'anisotropy_scaling_y', 'anisotropy_scaling_z', 'anisotropy_angle_z'}
    assert 1.8 < np.median([fit['anisotropy_scaling_z'] for fit in fits]) < 4.5
    assert np.median([fit['anisotropy_scaling_y'] for fit in fits]) < 1.8


def test_estimate_finds_little_anisotropy_in_an_isotropic_field():
    fits = [
        estimate_anisotropy(*random_field(350, seed), 'gaussian', nlags=12) for seed in range(5)
    ]
    scaling = [fit['anisotropy_scaling'] for fit in fits]
    assert max(scaling) < 1.8
    assert np.median(scaling) < 1.5


@pytest.mark.parametrize('model', ['linear', 'power', 'custom'])
def test_estimate_needs_a_bounded_model(model):
    points, values = random_field(40, 0)
    with pytest.raises(ValueError, match="variogram_model='spherical'"):
        estimate_anisotropy(points, values, model, nlags=6)


def test_estimate_explains_too_few_pairs_per_direction():
    # Samples on a line leave most horizontal directions without any pairs.
    rng = np.random.default_rng(0)
    along = rng.uniform(0, 100, 40)
    angle = np.radians(11.25)
    points = np.column_stack([along * np.cos(angle), along * np.sin(angle)])
    with pytest.raises(ValueError, match='Too few sample pairs per direction'):
        estimate_anisotropy(points, rng.standard_normal(40), 'spherical', nlags=6)


def test_estimate_needs_vertical_pairs_in_3d():
    points, values = random_field(60, 0, extent=(100.0, 100.0, 1e-3), scales=(8.0, 8.0, 8.0))
    with pytest.raises(ValueError, match='vertical'):
        estimate_anisotropy(points, values, 'spherical', nlags=6)


@pytest.fixture
def anisotropic2d() -> pv.PolyData:
    """Samples of an exponential field with correlation lengths 24 and 4 along 30 degrees."""
    return as_samples(*random_field(350, 0, covariance='exponential', angle=30, scales=(24, 4)))


@pytest.fixture
def layered3d() -> pv.PolyData:
    return as_samples(*random_field(400, 2, scales=(15.0, 15.0, 5.0), extent=(80.0, 80.0, 50.0)))


def test_fit_auto_applies_the_estimated_anisotropy_in_2d(anisotropic2d):
    model = anisotropic2d.krige.fit(
        'value', variogram_model='exponential', nlags=12, anisotropy='auto'
    )
    estimate = model.anisotropy
    assert set(estimate) == {'anisotropy_scaling', 'anisotropy_angle'}
    assert angle_error(estimate['anisotropy_angle'], 30.0) < 20.0
    assert estimate['anisotropy_scaling'] > 1.5
    assert model.instance.anisotropy_scaling == estimate['anisotropy_scaling']
    assert model.instance.anisotropy_angle == estimate['anisotropy_angle']


def test_fit_auto_applies_the_estimated_anisotropy_in_3d(layered3d):
    model = layered3d.krige.fit('value', variogram_model='gaussian', nlags=12, anisotropy='auto')
    estimate = model.anisotropy
    assert set(estimate) == {
        'anisotropy_scaling_y',
        'anisotropy_scaling_z',
        'anisotropy_angle_x',
        'anisotropy_angle_y',
        'anisotropy_angle_z',
    }
    assert estimate['anisotropy_scaling_z'] > 1.5
    assert estimate['anisotropy_angle_x'] == estimate['anisotropy_angle_y'] == 0.0
    assert model.instance.anisotropy_scaling_z == estimate['anisotropy_scaling_z']


def test_anisotropy_property_reports_given_and_default_anisotropy(samples2d, samples3d):
    assert samples2d.krige.fit('elevation').anisotropy == {
        'anisotropy_scaling': 1.0,
        'anisotropy_angle': 0.0,
    }
    model = samples3d.krige.fit('temperature', anisotropy_scaling_z=2.0, anisotropy_angle_z=40.0)
    assert model.anisotropy['anisotropy_scaling_z'] == 2.0
    assert model.anisotropy['anisotropy_angle_z'] == 40.0


def test_estimated_anisotropy_lowers_the_cross_validation_error():
    # Fitting the variogram to the short lags, as the docs advise, gives the clearest gain.
    for seed in range(3):
        samples = as_samples(
            *random_field(350, seed, covariance='exponential', angle=30, scales=(24, 4))
        )
        options = {'variogram_model': 'exponential', 'nlags': 12, 'max_lag': 35.0}
        isotropic = samples.krige.fit('value', **options).cross_validate(5)
        estimated = samples.krige.fit('value', anisotropy='auto', **options).cross_validate(5)
        assert estimated.rmse < isotropic.rmse


def test_cross_validate_estimates_anisotropy_in_every_fold(anisotropic2d):
    model = anisotropic2d.krige.fit('value', variogram_model='spherical', anisotropy='auto')
    held_out = model.cross_validate(3).rmse
    # Reusing the estimate from all samples would let the held-out samples shape it.
    reused = anisotropic2d.krige.fit('value', variogram_model='spherical', **model.anisotropy)
    assert np.isfinite(held_out)
    assert held_out != reused.cross_validate(3).rmse


def test_with_variogram_keeps_the_estimated_anisotropy(anisotropic2d):
    model = anisotropic2d.krige.fit('value', variogram_model='exponential', anisotropy='auto')
    refit = model.with_variogram('spherical')
    assert refit.anisotropy == model.anisotropy
    assert model.anisotropy['anisotropy_scaling'] > 1.0
    assert np.isfinite(refit.cross_validate(3).rmse)


def test_with_variogram_takes_anisotropy_overrides(anisotropic2d):
    model = anisotropic2d.krige.fit('value', variogram_model='exponential', anisotropy='auto')
    refit = model.with_variogram('spherical', anisotropy_scaling=1.0, anisotropy_angle=0.0)
    assert refit.anisotropy == {'anisotropy_scaling': 1.0, 'anisotropy_angle': 0.0}


def test_with_variogram_to_a_gstools_model_uses_its_anisotropy(anisotropic2d):
    gs = pytest.importorskip('gstools')
    model = anisotropic2d.krige.fit('value', variogram_model='exponential', anisotropy='auto')
    refit = model.with_variogram(gs.Exponential(dim=2, len_scale=10.0))
    assert refit.anisotropy == {'anisotropy_scaling': 1.0, 'anisotropy_angle': 0.0}


def test_auto_anisotropy_uses_the_drift_residuals_with_detrend(anisotropic2d):
    points = anisotropic2d.points
    anisotropic2d['value'] = anisotropic2d['value'] + 0.08 * points[:, 0] - 0.05 * points[:, 1]
    options = {
        'variogram_model': 'exponential',
        'method': 'universal',
        'drift_terms': ['regional_linear'],
        'nlags': 12,
        'max_lag': 35.0,
        'anisotropy': 'auto',
    }
    detrended = anisotropic2d.krige.fit('value', detrend=True, **options)
    raw = anisotropic2d.krige.fit('value', **options)
    assert angle_error(detrended.anisotropy['anisotropy_angle'], 30.0) < 20.0
    assert angle_error(raw.anisotropy['anisotropy_angle'], 30.0) > 20.0


def test_add_anisotropy_ellipsoid_draws_an_estimated_anisotropy(layered3d):
    model = layered3d.krige.fit('value', variogram_model='gaussian', nlags=12, anisotropy='auto')
    pl = pv.Plotter(off_screen=True)
    actor = pl.krige.add_anisotropy_ellipsoid(model)
    xmin, xmax, _, _, zmin, zmax = actor.mapper.dataset.bounds
    assert (xmax - xmin) > (zmax - zmin)
    pl.show()


def test_auto_anisotropy_needs_a_bounded_model(anisotropic2d):
    with pytest.raises(ValueError, match="variogram_model='spherical'"):
        anisotropic2d.krige.fit('value', anisotropy='auto')


def test_auto_anisotropy_rejects_a_gstools_model(anisotropic2d):
    gs = pytest.importorskip('gstools')
    with pytest.raises(ValueError, match='CovModel'):
        anisotropic2d.krige.fit(
            'value', variogram_model=gs.Exponential(dim=2, len_scale=10.0), anisotropy='auto'
        )


def test_auto_anisotropy_rejects_a_custom_variogram(anisotropic2d):
    with pytest.raises(ValueError, match='named PyKrige variogram model'):
        anisotropic2d.krige.fit(
            'value',
            variogram_model='custom',
            variogram_function=lambda parameters, distance: parameters[0] * distance,
            variogram_parameters=[1.0],
            anisotropy='auto',
        )


def test_auto_anisotropy_rejects_geographic_coordinates(anisotropic2d):
    with pytest.raises(ValueError, match='geographic'):
        anisotropic2d.krige.fit(
            'value',
            variogram_model='spherical',
            coordinates_type='geographic',
            anisotropy='auto',
        )


def test_auto_anisotropy_conflicts_with_given_anisotropy(anisotropic2d):
    with pytest.raises(ValueError, match='anisotropy_scaling'):
        anisotropic2d.krige.fit(
            'value', variogram_model='spherical', anisotropy='auto', anisotropy_scaling=2.0
        )


@pytest.mark.parametrize('value', ['manual', True, 2.0])
def test_anisotropy_option_only_takes_auto(anisotropic2d, value):
    with pytest.raises(ValueError, match="None or 'auto'"):
        anisotropic2d.krige.fit('value', variogram_model='spherical', anisotropy=value)


def test_anisotropy_none_means_no_estimate(anisotropic2d):
    model = anisotropic2d.krige.fit('value', variogram_model='spherical', anisotropy=None)
    assert model.anisotropy == {'anisotropy_scaling': 1.0, 'anisotropy_angle': 0.0}


def test_grid_search_cross_validate_rejects_auto_anisotropy(anisotropic2d):
    pytest.importorskip('sklearn')
    with pytest.raises(ValueError, match='anisotropy'):
        anisotropic2d.krige.cross_validate(
            'value', {'variogram_model': ['spherical']}, anisotropy='auto'
        )
