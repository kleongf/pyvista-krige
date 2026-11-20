"""Suggest where to take the next samples on PyVista's saddle surface.

Kriges the 16 sparse points onto the surface, then asks ``suggest_samples`` for
the five places where the kriging error is largest. Picks are made one at a time,
and each pick is added as a pseudo-sample before the next, so they do not all
land in the same corner. ``min_spacing`` keeps them 10 units apart. The right
panel adds the five picks as samples with the same variogram and shows the
standard deviation that is left.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

surface = examples.download_saddle_surface()
points = examples.download_sparse_points()
points['elevation'] = points.points[:, 2]

model = points.krige.fit('elevation', dim=2, variogram_model='gaussian')
before = model.predict(surface, std=True)
picks = model.suggest_samples(surface, n=5, min_spacing=10.0)
print(picks['pick order'], picks['std before pick'])

# Add the picks as samples, with the estimate as their value, to see what they would buy.
estimate = model.predict(picks)['elevation']
extended = pv.PolyData(np.vstack([points.points, picks.points]))
extended['elevation'] = np.concatenate([points['elevation'], estimate])
parameters = dict(zip(('psill', 'range', 'nugget'), model.variogram.parameters, strict=True))
after = extended.krige.fit(
    'elevation', dim=2, variogram_model='gaussian', variogram_parameters=parameters
).predict(surface, std=True)

clim = (0.0, before['elevation std'].max())
sargs = {'point_size': 18.0, 'render_points_as_spheres': True}
labels = [str(order) for order in picks['pick order']]
pl = pv.Plotter(shape=(1, 2), window_size=(1400, 600))
pl.add_text('Std before, with the suggested samples', font_size=12)
pl.add_mesh(before, scalars='elevation std', cmap='magma', clim=clim, show_scalar_bar=False)
pl.add_mesh(points, color='w', **sargs)
pl.add_mesh(picks, color='cyan', **sargs)
pl.add_point_labels(
    picks, labels, font_size=24, shape=None, show_points=False, always_visible=True
)
pl.subplot(0, 1)
pl.add_text('Std after adding them', font_size=12)
pl.add_mesh(after, scalars='elevation std', cmap='magma', clim=clim, show_scalar_bar=False)
pl.add_mesh(points, color='w', **sargs)
pl.add_mesh(picks, color='cyan', **sargs)
pl.link_views()
pl.view_xy()
pl.show()
