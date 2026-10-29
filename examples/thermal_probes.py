"""Krige subsurface temperature from scattered thermal probes.

The same data as PyVista's interpolation example, kriged in 3D with an
uncertainty estimate and the fitted variogram shown as an inset chart.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

probes = examples.download_thermal_probes()

grid = pv.ImageData(
    dimensions=(30, 38, 50), spacing=(500, 500, 100), origin=(329700, 4252600, -2700)
)

model = probes.krige.fit('temperature (C)', variogram_model='spherical', nlags=12)
kriged = model.predict(grid, n_closest_points=24)
print(model.variogram)

dargs = {'cmap': 'coolwarm', 'clim': [0, 300], 'scalars': 'temperature (C)'}
cpos = pv.CameraPosition(
    position=(364281.0, 4285326.0, 14093.0),
    focal_point=(337749.0, 4261154.0, -637.0),
    viewup=(-0.2963, -0.2384, 0.9249),
)

pl = pv.Plotter(shape=(1, 2), window_size=(1600, 700))
pl.add_mesh(kriged.contour(5, scalars='temperature (C)'), opacity=0.5, **dargs)
pl.add_mesh(probes, render_points_as_spheres=True, point_size=10, **dargs)
pl.krige.add_variogram(model, loc=(0.62, 0.62), size=(0.36, 0.36))
pl.subplot(0, 1)
pl.add_mesh(kriged.slice_orthogonal(), scalars='temperature (C) variance', cmap='viridis')
pl.add_mesh(probes, color='k', render_points_as_spheres=True, point_size=6)
pl.link_views()
pl.show(cpos=cpos)
