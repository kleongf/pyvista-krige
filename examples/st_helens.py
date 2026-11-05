"""Rebuild Mount St. Helens from 400 spot elevations.

Samples the real DEM at random locations, picks a variogram by 5-fold
cross-validation, kriges the elevation back onto the full grid, and compares
the result with the truth. The variogram is fitted to 20 lags up to 7 km
(``max_lag``); PyKrige's default of 6 lags over the full extent mostly sees
the cone's trend. The largest errors sit along the DEM border, where every
estimate is an extrapolation from samples on one side.
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

scores = {}
for variogram_model in ['linear', 'spherical', 'exponential', 'gaussian']:
    candidate = samples.krige.fit(
        'Elevation', variogram_model=variogram_model, nlags=20, max_lag=7000.0
    )
    scores[variogram_model] = candidate.cross_validate(5).rmse
    print(f'{variogram_model:12s} 5-fold RMSE {scores[variogram_model]:6.1f} m')
best = min(scores, key=scores.get)
model = samples.krige.fit('Elevation', variogram_model=best, nlags=20, max_lag=7000.0)
# Kriged in chunks against all 400 samples, without a moving window.
kriged = model.predict(dem, std=True)
kriged['Error'] = kriged['Elevation'] - dem['Elevation']

factor = 2.0
clim = dem.get_data_range('Elevation')
lifted = samples.copy()  # sit the markers just above the warped surface
lifted.points[:, 2] += factor * samples['Elevation'] + 60.0


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
    }


pl = pv.Plotter(shape=(2, 3), window_size=(2100, 1300))
pl.add_text('True DEM', font_size=12)
pl.add_mesh(
    dem.warp_by_scalar('Elevation', factor=factor),
    cmap='gist_earth',
    clim=clim,
    scalar_bar_args=bar('Elevation (m)'),
)
pl.reset_camera()
pl.camera.zoom(0.8)  # leave room for the scalar bar below the terrain
pl.camera.focal_point = (*pl.camera.focal_point[:2], pl.camera.focal_point[2] - 1500.0)
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
pl.add_text(f'Variogram: {best}, 5-fold RMSE {scores[best]:.0f} m', font_size=12)
pl.krige.add_variogram(model, loc=(0.08, 0.08), size=(0.88, 0.8))
pl.subplot(1, 0)
rmse = np.sqrt(np.mean(kriged['Error'] ** 2))
pl.add_text(f'Error (m), RMSE {rmse:.0f} m', font_size=12)
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
