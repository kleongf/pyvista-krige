from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv


def test_add_variogram_adds_chart_with_experimental_points_and_model_curve(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    chart = pl.krige.add_variogram(model, loc=(0.6, 0.05), size=(0.38, 0.4))
    assert isinstance(chart, pv.Chart2D)
    assert len(list(chart.plots('scatter'))) == 1
    assert len(list(chart.plots('line'))) == 1
    pl.show()


def test_add_uncertainty_handles_variances_far_above_one(samples2d):
    samples2d['elevation'] = 1000.0 * samples2d['elevation']  # variance in the 1e5 range
    image = samples2d.krige.to_image('elevation', dimensions=(20, 20), padding=0.5)
    assert image['elevation variance'].max() > 1.0

    def render(add):
        pl = pv.Plotter(window_size=(200, 200))
        add(pl)
        pl.view_xy()
        return pl.screenshot(return_img=True)

    faded = render(lambda pl: pl.krige.add_uncertainty(image, 'elevation'))
    opaque = render(lambda pl: pl.add_mesh(image, scalars='elevation'))
    empty = render(lambda pl: None)
    assert not np.array_equal(faded, opaque)
    assert not np.array_equal(faded, empty)  # the least certain areas stay faintly visible
    no_fade = render(lambda pl: pl.krige.add_uncertainty(image, 'elevation', min_opacity=1.0))
    np.testing.assert_array_equal(no_fade, opaque)


def test_add_variogram_draws_one_series_per_variogram(samples3d):
    horizontal = samples3d.krige.variogram('temperature', direction='horizontal', nlags=5)
    vertical = samples3d.krige.variogram('temperature', direction='vertical', nlags=5)
    model = samples3d.krige.fit('temperature', variogram_model='spherical')
    pl = pv.Plotter()
    chart = pl.krige.add_variogram([model, horizontal, vertical])
    assert len(list(chart.plots('scatter'))) == 3
    assert len(list(chart.plots('line'))) == 1
    pl.show()


def test_exceedance_probability_follows_normal_tail(samples3d):
    from scipy.stats import norm

    import pyvista_krige

    image = samples3d.krige.to_image('temperature', dimensions=(8, 8, 4))
    prob = pyvista_krige.exceedance(image, 'temperature', 5.0)
    name = 'P(temperature > 5)'
    std = np.sqrt(np.clip(image['temperature variance'], 0, None))
    expected = norm.sf((5.0 - image['temperature']) / np.where(std > 0, std, np.inf))
    expected = np.where(std > 0, expected, (image['temperature'] > 5.0).astype(float))
    np.testing.assert_allclose(prob[name], expected)
    below = pyvista_krige.exceedance(image, 'temperature', 5.0, below=True)
    np.testing.assert_allclose(below['P(temperature < 5)'], 1.0 - expected)
    pl = pv.Plotter()
    actor = pl.krige.add_exceedance(image, 'temperature', 5.0, probability=0.9)
    assert isinstance(actor, pv.Actor)
    pl.show()


def test_add_residuals_scales_spheres_by_z_score(samples3d):
    model = samples3d.krige.fit('temperature', variogram_model='linear')
    holdout = model.cross_validate(folds=3)
    pl = pv.Plotter()
    actor = pl.krige.add_residuals(holdout)
    glyphs = actor.mapper.dataset
    assert glyphs.n_points > samples3d.n_points  # one sphere per sample
    error = holdout.samples['temperature error']
    lim = np.abs(error).max()
    assert actor.mapper.scalar_range == pytest.approx((-lim, lim))
    pl.show()


def test_add_anisotropy_ellipsoid_matches_variogram_range(samples3d):
    model = samples3d.krige.fit(
        'temperature',
        variogram_model='spherical',
        variogram_parameters={'sill': 2.0, 'range': 4.0, 'nugget': 0.0},
        anisotropy_scaling_y=2.0,
        anisotropy_scaling_z=4.0,
        anisotropy_angle_z=90.0,
    )
    pl = pv.Plotter()
    actor = pl.krige.add_anisotropy_ellipsoid(model, center=(0.0, 0.0, 0.0))
    xmin, xmax, ymin, ymax, zmin, zmax = actor.mapper.dataset.bounds
    # Rotated 90 degrees about z, the major axis lies along y and the minor along x.
    assert (ymax - ymin) / 2 == pytest.approx(4.0, rel=1e-2)
    assert (xmax - xmin) / 2 == pytest.approx(2.0, rel=1e-2)
    assert (zmax - zmin) / 2 == pytest.approx(1.0, rel=1e-2)
    pl.show()


def test_add_anisotropy_ellipsoid_needs_a_bounded_model(samples3d):
    model = samples3d.krige.fit('temperature', variogram_model='linear')
    with pytest.raises(ValueError, match='range'):
        pv.Plotter().krige.add_anisotropy_ellipsoid(model)
