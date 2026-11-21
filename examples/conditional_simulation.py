"""Simulate Mount St. Helens from 400 spot elevations.

Kriging gives the single smoothest surface through the samples, which is too
smooth to be a real landscape. ``model.simulate`` draws equally likely surfaces
that pass through the same samples and vary as much as the data do. The
elevation is skewed, with a few high points on the cone, so the model is
fitted in normal score space and each realization is back-transformed to
metres.

The fraction of 100 realizations above 2000 m is compared with the exceedance
probability that ``pyvista_krige.exceedance`` reads from the kriging variance.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

import pyvista_krige

dem = examples.download_st_helens()
rng = np.random.default_rng(42)
picked = rng.choice(dem.n_points, size=400, replace=False)
samples = pv.PolyData(dem.points[picked])
samples['Elevation'] = dem['Elevation'][picked]

model = samples.krige.fit(
    'Elevation', variogram_model='exponential', nlags=20, max_lag=7000.0, transform='normal_score'
)
# Every fourth DEM node, about 9,500 points; the simulation visits them one by one.
grid = dem.extract_subset((0, dem.dimensions[0] - 1, 0, dem.dimensions[1] - 1, 0, 0), (4, 4, 1))
grid.clear_data()
threshold = 2000.0
kriged = pyvista_krige.exceedance(model.predict(grid), 'Elevation', threshold, model=model)
simulated = model.simulate(grid, n=2, seed=0)
probability = model.simulate(grid, n=100, seed=1, exceed=threshold)
name = f'P(Elevation > {threshold:g})'
difference = np.abs(probability[name] - kriged[name])
print(f'Largest difference in {name}: {difference.max():.2f}, mean {difference.mean():.3f}')

clim = dem.get_data_range('Elevation')
panels = [
    (kriged, 'Elevation', 'Kriged estimate', clim, 'gist_earth'),
    (simulated, 'Elevation realization 0', 'Realization 0', clim, 'gist_earth'),
    (simulated, 'Elevation realization 1', 'Realization 1', clim, 'gist_earth'),
    (kriged, name, f'{name}, kriging variance', (0, 1), 'viridis'),
    (probability, name, f'{name}, 100 realizations', (0, 1), 'viridis'),
]
pl = pv.Plotter(shape=(1, 5), window_size=(1800, 560))
for column, (mesh, scalars, title, limits, cmap) in enumerate(panels):
    pl.subplot(0, column)
    pl.add_text(title, font_size=10)
    pl.add_mesh(mesh, scalars=scalars, clim=limits, cmap=cmap, show_scalar_bar=False)
    pl.add_mesh(samples, color='w', point_size=3.0, render_points_as_spheres=True)
    pl.view_xy()
    pl.camera.zoom(1.5)
pl.link_views()
pl.show()
