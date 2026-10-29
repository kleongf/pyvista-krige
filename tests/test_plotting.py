from __future__ import annotations

import numpy as np
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
