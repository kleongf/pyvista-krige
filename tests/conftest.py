from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

pv.OFF_SCREEN = True


@pytest.fixture
def samples2d() -> pv.PolyData:
    """Scattered points on the z=0 plane with a smooth 'elevation' field."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(40, 2))
    points = np.column_stack([xy, np.zeros(len(xy))])
    samples = pv.PolyData(points)
    samples['elevation'] = np.sin(xy[:, 0] / 3) + 0.1 * xy[:, 1]
    return samples


@pytest.fixture
def samples3d() -> pv.PolyData:
    """Scattered points in a 10 x 10 x 5 box with a smooth 'temperature' field."""
    rng = np.random.default_rng(1)
    points = rng.uniform((0, 0, 0), (10, 10, 5), size=(50, 3))
    samples = pv.PolyData(points)
    samples['temperature'] = points[:, 0] + np.cos(points[:, 1] / 2) - 0.5 * points[:, 2]
    return samples
