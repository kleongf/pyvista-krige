from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from pyvista import _vtk

import pyvista_krige


@pytest.fixture
def preview() -> pv.ImageData:
    return pv.ImageData(dimensions=(12, 12, 1), spacing=(1.0, 1.0, 1.0))


def drag(slider, value):
    """Move a slider the way a mouse drag does and fire its interaction callback."""
    slider.GetRepresentation().SetValue(value)
    slider.InvokeEvent(_vtk.vtkCommand.InteractionEvent)


def sliders_of(pl):
    """Return the plotter's slider widgets, whichever PyVista version it is."""
    return getattr(pl, 'widgets', pl).slider_widgets


def slider_state(pl):
    """Return ``{title: (value, low, high)}`` of the plotter's sliders."""
    state = {}
    for slider in sliders_of(pl):
        rep = slider.GetRepresentation()
        state[rep.GetTitleText()] = (
            rep.GetValue(),
            rep.GetMinimumValue(),
            rep.GetMaximumValue(),
        )
    return state


def test_bounded_models_get_sill_range_and_nugget_sliders(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    state = slider_state(pl)
    assert list(state) == ['Partial sill', 'Range', 'Nugget']
    psill, rng, nugget = model.variogram.parameters
    assert state['Partial sill'][0] == pytest.approx(psill)
    assert state['Range'][0] == pytest.approx(rng)
    assert state['Nugget'][0] == pytest.approx(nugget)
    pl.close()


@pytest.mark.parametrize(
    ('name', 'titles'),
    [
        ('linear', ['Slope', 'Nugget']),
        ('power', ['Scale', 'Exponent', 'Nugget']),
    ],
)
def test_unbounded_models_get_sliders_for_their_own_parameters(samples2d, preview, name, titles):
    model = samples2d.krige.fit('elevation', variogram_model=name)
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    state = slider_state(pl)
    assert list(state) == titles
    for title, parameter in zip(titles, model.variogram.parameters, strict=True):
        assert state[title][0] == pytest.approx(parameter)
    pl.close()


@pytest.mark.parametrize('name', ['spherical', 'exponential', 'gaussian', 'linear', 'power'])
def test_every_slider_handle_starts_inside_a_usable_range(samples2d, preview, name):
    # PyKrige fits a nugget of about 1e-16 and puts the range on the largest lag, so
    # scaling the range by the fitted value alone would leave no room to move.
    model = samples2d.krige.fit('elevation', variogram_model=name)
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    for title, (value, low, high) in slider_state(pl).items():
        assert low <= value <= high, title
        assert high > low, title
        if title == 'Range':
            assert low < value < high
        if title == 'Nugget':
            assert high > 1e-3 * model.variogram.semivariance.max()
    pl.close()


# With a nugget near 0, scaling the sill, slope or scale leaves the kriging weights, and so
# the estimate, unchanged and only moves the variance. Every other parameter moves both.
SCALES = {'Partial sill', 'Slope', 'Scale'}


def check_drag(pl, preview, mesh, k, title, fraction, expected):
    """Drag slider ``k`` and check the mesh, the reported parameters and the callback."""
    _, low, high = slider_state(pl)[title]
    target = low + fraction * (high - low)
    estimate = mesh['elevation'].copy()
    variance = mesh['elevation variance'].copy()
    drag(sliders_of(pl)[k], target)
    expected[k] = target
    tuned = pl.krige.variogram_widget_model
    np.testing.assert_allclose(tuned.variogram.parameters, expected)
    fresh = tuned.predict(preview)
    np.testing.assert_allclose(mesh['elevation'], fresh['elevation'])
    np.testing.assert_allclose(mesh['elevation variance'], fresh['elevation variance'])
    assert not np.allclose(mesh['elevation variance'], variance), title
    if title not in SCALES:
        assert not np.allclose(mesh['elevation'], estimate), title
    return tuned


def test_each_slider_updates_the_preview_and_the_reported_parameters(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    received = []
    pl = pv.Plotter()
    actor = pl.krige.add_variogram_widget(model, preview, callback=received.append)
    mesh = actor.mapper.dataset
    assert pl.krige.variogram_widget_model is model
    assert received == []  # creating the sliders is not a change
    expected = model.variogram.parameters.copy()
    for k, title in enumerate(['Partial sill', 'Range', 'Nugget']):
        tuned = check_drag(pl, preview, mesh, k, title, 0.4, expected)
        assert received[-1] is tuned
    assert len(received) == 3
    pl.close()


@pytest.mark.parametrize('name', ['linear', 'power'])
def test_unbounded_sliders_update_the_preview_and_the_reported_parameters(
    samples2d, preview, name
):
    model = samples2d.krige.fit('elevation', variogram_model=name)
    pl = pv.Plotter()
    actor = pl.krige.add_variogram_widget(model, preview)
    mesh = actor.mapper.dataset
    expected = model.variogram.parameters.copy()
    for k, title in enumerate(slider_state(pl)):
        tuned = check_drag(pl, preview, mesh, k, title, 0.7, expected)
        assert tuned.variogram.model_name == name
    pl.close()


def test_the_preview_can_be_colored_by_the_variance_to_tune_the_sill(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    actor = pl.krige.add_variogram_widget(model, preview, scalars='elevation variance')
    before = actor.mapper.scalar_range
    drag(sliders_of(pl)[0], slider_state(pl)['Partial sill'][2])
    after = actor.mapper.scalar_range
    assert after[1] > 1.5 * before[1]  # the color range follows the variance
    with pytest.raises(ValueError, match='nonexistent'):
        pv.Plotter().krige.add_variogram_widget(model, preview, scalars='nonexistent')
    pl.close()


def test_the_preview_mesh_is_updated_in_place_and_the_input_is_untouched(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    actor = pl.krige.add_variogram_widget(model, preview)
    mesh = actor.mapper.dataset
    variance = mesh['elevation variance'].copy()
    high = slider_state(pl)['Nugget'][2]
    drag(sliders_of(pl)[2], high)
    assert actor.mapper.dataset is mesh
    assert not np.allclose(mesh['elevation variance'], variance)
    assert preview.array_names == []
    pl.close()


def test_the_tuned_model_is_independent_of_the_fitted_one(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    fitted = model.variogram.parameters.copy()
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    low = slider_state(pl)['Range'][1]
    drag(sliders_of(pl)[1], low)
    np.testing.assert_array_equal(model.variogram.parameters, fitted)
    assert pl.krige.variogram_widget_model is not model
    pl.close()


def test_the_variogram_chart_follows_the_sliders(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    assert len(pl.renderer._charts) == 1
    chart = pl.renderer._charts._charts[0]
    (line,) = chart.plots('line')
    before = np.array(line.y)
    high = slider_state(pl)['Partial sill'][2]
    drag(sliders_of(pl)[0], high)
    after = np.array(line.y)
    assert after.max() > before.max()
    pl.close()


def test_the_chart_can_be_turned_off(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview, show_variogram=False)
    assert not pl.renderer._charts
    pl.close()


def test_the_property_is_empty_until_a_widget_is_added():
    assert pv.Plotter().krige.variogram_widget_model is None


def test_a_large_preview_warns(samples2d):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    big = pv.ImageData(dimensions=(230, 230, 1))
    assert big.n_points > 50_000
    pl = pv.Plotter()
    with pytest.warns(UserWarning, match='coarse'):
        pl.krige.add_variogram_widget(model, big)
    pl.close()


def test_a_coarse_preview_does_not_warn(samples2d, preview, recwarn):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    pl.close()


def test_custom_variograms_have_no_known_parameters_to_tune(samples2d, preview):
    model = samples2d.krige.fit(
        'elevation',
        variogram_model='custom',
        variogram_parameters=[1.0, 3.0],
        variogram_function=lambda p, d: p[0] * (1 - np.exp(-d / p[1])),
    )
    with pytest.raises(ValueError, match='custom'):
        pv.Plotter().krige.add_variogram_widget(model, preview)


def test_a_singular_variogram_keeps_the_last_good_model(samples2d, preview):
    model = samples2d.krige.fit('elevation', variogram_model='spherical')
    pl = pv.Plotter()
    pl.krige.add_variogram_widget(model, preview)
    sliders = sliders_of(pl)
    drag(sliders[2], 0.0)  # no nugget: a valid change
    good = pl.krige.variogram_widget_model
    # A partial sill of zero and no nugget makes the kriging matrix singular.
    sliders[0].GetRepresentation().SetMinimumValue(0.0)
    with pytest.warns(UserWarning, match='singular'):
        drag(sliders[0], 0.0)
    assert pl.krige.variogram_widget_model is good
    pl.close()


def test_the_package_exports_the_component_method():
    assert hasattr(pyvista_krige.KrigeComponent, 'add_variogram_widget')
