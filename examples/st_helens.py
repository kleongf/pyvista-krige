"""Rebuild Mount St. Helens from 400 spot elevations.

Samples the real DEM at random locations, picks a variogram by
cross-validation, kriges the elevation back onto the full grid, and compares
the result with the truth.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

dem = examples.download_st_helens()
rng = np.random.default_rng(42)
picked = rng.choice(dem.n_points, size=400, replace=False)
samples = pv.PolyData(dem.points[picked])
samples['Elevation'] = dem['Elevation'][picked]

search = samples.krige.cross_validate(
    'Elevation',
    {
        'variogram_model': ['linear', 'spherical', 'exponential', 'gaussian'],
        'weight': [False, True],
    },
)
print(f'best: {search.best_params}  R2 = {search.best_score:.3f}')
model = search.model
kriged = model.predict(dem, n_closest_points=24, backend='C', std=True)
kriged['Error'] = kriged['Elevation'] - dem['Elevation']

factor = 2.0
clim = dem.get_data_range('Elevation')
lifted = samples.copy()  # sit the markers just above the warped surface
lifted.points[:, 2] += factor * samples['Elevation'] + 60.0


def bar(title):
    return {'title': title, 'vertical': True, 'position_x': 0.84, 'position_y': 0.15}


pl = pv.Plotter(shape=(2, 3), window_size=(2100, 1300))
pl.add_text('True DEM', font_size=12)
pl.add_mesh(
    dem.warp_by_scalar('Elevation', factor=factor),
    cmap='gist_earth',
    clim=clim,
    scalar_bar_args=bar('Elevation (m)'),
)
pl.subplot(0, 1)
pl.add_text('Kriged from 400 samples', font_size=12)
pl.add_mesh(
    kriged.warp_by_scalar('Elevation', factor=factor),
    scalars='Elevation',
    cmap='gist_earth',
    clim=clim,
    show_scalar_bar=False,
)
pl.add_mesh(lifted, color='k', point_size=5, render_points_as_spheres=True)
pl.subplot(0, 2)
pl.add_text(
    f'Variogram: {search.best_params["variogram_model"]}, CV R2 = {search.best_score:.2f}',
    font_size=12,
)
pl.krige.add_variogram(model, loc=(0.08, 0.08), size=(0.88, 0.8))
pl.subplot(1, 0)
pl.add_text('Error (m)', font_size=12)
pl.add_mesh(
    kriged, scalars='Error', cmap='RdBu', clim=(-150, 150), scalar_bar_args=bar('Error (m)')
)
pl.add_mesh(samples, color='k', point_size=4, render_points_as_spheres=True)
pl.view_xy()
pl.camera.zoom(0.85)
pl.subplot(1, 1)
pl.add_text('Kriging standard deviation (m)', font_size=12)
pl.add_mesh(kriged, scalars='Elevation std', cmap='magma', scalar_bar_args=bar('Std (m)'))
pl.add_mesh(samples, color='w', point_size=4, render_points_as_spheres=True)
pl.view_xy()
pl.camera.zoom(0.85)
pl.subplot(1, 2)
pl.add_text('add_uncertainty: fades where std is high', font_size=12)
pl.krige.add_uncertainty(
    kriged.warp_by_scalar('Elevation', factor=factor),
    'Elevation',
    cmap='gist_earth',
    clim=clim,
    show_scalar_bar=False,
)
pl.show()
