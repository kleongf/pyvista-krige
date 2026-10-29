"""Compare kriging with PyVista's Gaussian-kernel interpolation.

Uses the sparse points and saddle surface from PyVista's interpolation
example. Kriging honors every sample exactly and reports where it is unsure.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

surface = examples.download_saddle_surface()
points = examples.download_sparse_points()

gaussian = surface.interpolate(points, radius=12.0)
# With only 16 samples, a gaussian fit collapses to a pure nugget; spherical holds up.
model = points.krige.fit('val', variogram_model='spherical')
kriged = model.predict(surface, std=True)

sargs = {'point_size': 18.0, 'render_points_as_spheres': True}
clim = points.get_data_range('val')
pl = pv.Plotter(shape=(1, 3), window_size=(1800, 600))
pl.add_text('Gaussian kernel (interpolate)', font_size=12)
pl.add_mesh(gaussian, scalars='val', clim=clim, cmap='viridis')
pl.add_mesh(points, scalars='val', clim=clim, cmap='viridis', **sargs)
pl.subplot(0, 1)
pl.add_text('Kriging estimate', font_size=12)
pl.add_mesh(kriged, scalars='val', clim=clim, cmap='viridis')
pl.add_mesh(points, scalars='val', clim=clim, cmap='viridis', **sargs)
pl.subplot(0, 2)
pl.add_text('Kriging standard deviation', font_size=12)
pl.add_mesh(kriged, scalars='val std', cmap='magma')
pl.add_mesh(points, color='w', **sargs)
pl.link_views()
pl.camera_position = 'iso'
pl.show()
