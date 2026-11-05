"""Compare kriging with PyVista's Gaussian-kernel interpolation.

Uses the 16 sparse points and saddle surface from PyVista's interpolation
example, interpolating the elevation of the points back onto the surface so
both methods can be scored against the truth. The variogram is chosen by
leave-one-out cross-validation, and kriging also reports where it is unsure.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

surface = examples.download_saddle_surface()
points = examples.download_sparse_points()
points['elevation'] = points.points[:, 2]
surface['true elevation'] = surface.points[:, 2]


def rmse(values):
    return np.sqrt(np.mean((values - surface['true elevation']) ** 2))


gaussian = surface.interpolate(points, radius=12.0)

# Pick the variogram with the smallest leave-one-out error.
scores = {}
for variogram_model in ['linear', 'power', 'gaussian', 'spherical', 'exponential']:
    candidate = points.krige.fit('elevation', dim=2, variogram_model=variogram_model)
    scores[variogram_model] = candidate.cross_validate('loo').rmse
    print(f'{variogram_model:12s} leave-one-out RMSE {scores[variogram_model]:.2f}')
best = min(scores, key=scores.get)
model = points.krige.fit('elevation', dim=2, variogram_model=best)
kriged = model.predict(surface, std=True)

sargs = {'point_size': 18.0, 'render_points_as_spheres': True}
clim = surface.get_data_range('true elevation')
dargs = {'clim': clim, 'cmap': 'viridis', 'show_scalar_bar': False}
pl = pv.Plotter(shape=(1, 4), window_size=(2200, 600))
pl.add_text('True elevation', font_size=12)
pl.add_mesh(surface, scalars='true elevation', **dargs)
pl.add_mesh(points, scalars='elevation', **sargs, **dargs)
pl.subplot(0, 1)
pl.add_text(f'Gaussian kernel, RMSE {rmse(gaussian["elevation"]):.2f}', font_size=12)
pl.add_mesh(gaussian, scalars='elevation', **dargs)
pl.add_mesh(points, scalars='elevation', **sargs, **dargs)
pl.subplot(0, 2)
pl.add_text(f'Kriging ({best} variogram), RMSE {rmse(kriged["elevation"]):.2f}', font_size=12)
pl.add_mesh(kriged, scalars='elevation', **dargs)
pl.add_mesh(points, scalars='elevation', **sargs, **dargs)
pl.subplot(0, 3)
pl.add_text('Kriging standard deviation', font_size=12)
pl.add_mesh(kriged, scalars='elevation std', cmap='magma')
pl.add_mesh(points, color='w', **sargs)
pl.link_views()
pl.camera_position = 'iso'
pl.show()
