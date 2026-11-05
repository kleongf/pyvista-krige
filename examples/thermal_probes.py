"""Krige subsurface temperature from scattered thermal probes.

The same data as PyVista's interpolation example, kriged in 3D. Temperature
rises steadily with depth, so ordinary kriging, which reverts to the mean away
from the probes, bends the isotherms back at the edges of the volume. Universal
kriging with a depth drift follows the gradient instead. Holding out the 69
deepest probes and predicting them from the rest shows the difference.

The variogram is fitted only to lags up to 9 km (``max_lag``), because the
long lags reflect the trend rather than spatial correlation. The standard
deviation slices are cut off 2.5 km from the nearest probe (``max_distance``).
Far from every probe near the surface, the linear depth drift extrapolates to
slightly below 0 °C.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

probes = examples.download_thermal_probes()
probes['depth'] = probes.points[:, 2]
name = 'temperature (C)'

grid = pv.ImageData(
    dimensions=(30, 38, 50), spacing=(500, 500, 100), origin=(329700, 4252600, -2700)
)
grid['depth'] = grid.points[:, 2]

ordinary = {'variogram_model': 'spherical', 'nlags': 12}
universal = {
    'method': 'universal',
    'specified_drift': ['depth'],  # a linear drift in z only
    'variogram_model': 'exponential',
    'nlags': 12,
    'max_lag': 9000.0,
}

# Score both models on the deep probes, then on random 10-fold cross-validation.
deep = probes.points[:, 2] < -1000
shallow = probes.extract_points(~deep, adjacent_cells=False, include_cells=False)
held_out = probes.extract_points(deep, adjacent_cells=False, include_cells=False)
for label, kwargs, window in [('ordinary', ordinary, 24), ('universal', universal, None)]:
    fitted = shallow.krige.fit(name, **kwargs)
    estimate = fitted.predict(held_out, variance=False, n_closest_points=window)[name]
    deep_rmse = np.sqrt(np.mean((estimate - held_out[name]) ** 2))
    holdout = probes.krige.fit(name, **kwargs).cross_validate(10, n_closest_points=window)
    print(f'{label:9s}  deep-probe RMSE {deep_rmse:5.1f} C   10-fold RMSE {holdout.rmse:5.1f} C')

model = probes.krige.fit(name, **universal)
kriged = model.predict(grid, std=True)
near = model.predict(grid, std=True, max_distance=2500.0)
holdout = model.cross_validate(10)
print(model.variogram)

# Variograms of the depth-trend residuals, to check for anisotropy left after the drift.
residual = {'max_lag': 9000.0, 'detrend': True, 'specified_drift': ['depth']}
horizontal = probes.krige.variogram(name, direction='horizontal', **residual)
vertical = probes.krige.variogram(name, direction='vertical', **residual)


def bar(title):
    """Keep scalar bars in a strip along the bottom edge, below the meshes."""
    return {
        'title': title,
        'vertical': False,
        'position_x': 0.2,
        'position_y': 0.03,
        'width': 0.6,
        'height': 0.08,
        'fmt': '%.0f',
        'title_font_size': 14,
        'label_font_size': 12,
    }


cpos = pv.CameraPosition(
    position=(364281.0, 4285326.0, 14093.0),
    focal_point=(337749.0, 4261154.0, -637.0),
    viewup=(-0.2963, -0.2384, 0.9249),
)
outline = grid.outline()
dargs = {'cmap': 'coolwarm', 'clim': [0, 300], 'scalar_bar_args': bar('Temperature (C)')}
views = [(0, 0), (0, 1), (1, 0), (1, 1)]

pl = pv.Plotter(shape=(2, 3), window_size=(2100, 1200))
pl.add_text('Isotherms: universal kriging with a depth drift', font_size=10)
isotherms = kriged.contour([50, 100, 150, 200, 250], scalars=name)
pl.add_mesh(isotherms, scalars=name, opacity=0.6, **dargs)
pl.add_mesh(probes, scalars=name, render_points_as_spheres=True, point_size=8, **dargs)
pl.add_mesh(outline, color='k')

pl.subplot(0, 1)
pl.add_text('Kriging std within 2.5 km of a probe', font_size=10)
pl.add_mesh(
    near.slice_orthogonal(),
    scalars=f'{name} std',
    cmap='viridis',
    nan_opacity=0.0,
    scalar_bar_args=bar('Std (C)'),
)
pl.add_mesh(probes, color='k', render_points_as_spheres=True, point_size=5)
pl.add_mesh(outline, color='k')

pl.subplot(0, 2)
pl.add_text('Fitted variogram (lags up to 9 km)', font_size=10)
pl.krige.add_variogram(model, loc=(0.1, 0.08), size=(0.86, 0.82))

pl.subplot(1, 0)
pl.add_text('Above 150 C: 50 % (outer) and 90 % (inner) likely', font_size=10)
pl.krige.add_exceedance(kriged, name, 150.0, probability=0.5, color='orange', opacity=0.3)
pl.krige.add_exceedance(kriged, name, 150.0, probability=0.9, color='firebrick')
pl.add_mesh(probes, color='k', render_points_as_spheres=True, point_size=5)
pl.add_mesh(outline, color='k')

pl.subplot(1, 1)
pl.add_text(f'10-fold hold-out errors, RMSE {holdout.rmse:.1f} C', font_size=10)
pl.krige.add_residuals(holdout, scalar_bar_args=bar('Error (C)'))
pl.add_mesh(outline, color='k')

pl.subplot(1, 2)
pl.add_text('Detrended directional variograms', font_size=10)
pl.krige.add_variogram([horizontal, vertical], loc=(0.1, 0.08), size=(0.86, 0.82))

for view in views:
    pl.subplot(*view)
    pl.camera_position = cpos
    pl.camera.zoom(0.72)
pl.show()
