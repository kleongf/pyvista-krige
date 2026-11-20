"""Cell declustering weights for clustered samples."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

WEIGHT_NAME = 'declustering weight'

# Irrational shifts, so the grid origins of successive offsets never line up.
_OFFSET_STEPS = np.sqrt([2.0, 3.0, 5.0]) % 1.0


@dataclass(frozen=True)
class DeclusterScan:
    """Declustered mean of the samples for each cell size tried.

    Attributes
    ----------
    cell_sizes : numpy.ndarray
        Cell size per scan step, shaped ``(n_sizes, dim)``.

    means : numpy.ndarray
        Declustered mean for each cell size.

    best : int
        Index of the chosen cell size.

    """

    cell_sizes: np.ndarray
    means: np.ndarray
    best: int

    @property
    def cell_size(self) -> tuple[float, ...]:
        """Chosen cell size, one value per axis."""
        return tuple(float(s) for s in self.cell_sizes[self.best])


def cell_vector(cell_size: float | tuple[float, ...], points: np.ndarray) -> np.ndarray:
    """Return ``cell_size`` as one positive size per axis.

    A number is a square cell for flat samples and a cube for samples with a
    spread in z. A tuple sets the axes, so its length is the dimension.
    """
    size = np.atleast_1d(np.asarray(cell_size, dtype=float))
    if size.ndim != 1 or size.size not in (1, 2, 3):
        msg = f'cell_size must be a number or a 2- or 3-tuple, got {cell_size!r}.'
        raise ValueError(msg)
    if size.size == 1:
        dim = 3 if np.ptp(points[:, 2]) > 0 else 2
        size = np.full(dim, size[0])
    if not (np.isfinite(size).all() and (size > 0).all()):
        msg = f'cell_size must be positive and finite, got {cell_size!r}.'
        raise ValueError(msg)
    return size


def declustering_weights(points: np.ndarray, cell: np.ndarray, n_offsets: int = 10) -> np.ndarray:
    """Return cell declustering weights that sum to the number of points.

    Each point gets 1 over the number of points in its cell. The weights are
    scaled to sum to the number of points, then averaged over ``n_offsets``
    shifts of the grid origin, so the result does not depend on where the grid
    starts. The grid is anchored at the minimum coordinate, so the sample order
    does not matter either.
    """
    if n_offsets < 1:
        msg = f'n_offsets must be at least 1, got {n_offsets!r}.'
        raise ValueError(msg)
    coords = points[:, : cell.size]
    n = len(coords)
    # Round off division noise, so a sample one cell size from the origin does not
    # fall just below the cell boundary and share a cell with its neighbor.
    scaled = np.round((coords - coords.min(axis=0)) / cell, 9)
    steps = _OFFSET_STEPS[: cell.size]
    total = np.zeros(n)
    for j in range(n_offsets):
        shifted = scaled + (j * steps) % 1.0
        cells = np.floor(shifted).astype(np.int64)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse.ravel()]
        total += weight * (n / weight.sum())
    return total / n_offsets


def declustered_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean of the finite ``values``."""
    finite = np.isfinite(values)
    return float(np.average(values[finite], weights=weights[finite]))


def scan_declustering(
    points: np.ndarray,
    values: np.ndarray,
    *,
    n_sizes: int = 20,
    n_offsets: int = 10,
    size_range: tuple[float, float] | None = None,
    aspect: tuple[float, ...] | None = None,
    maximize: bool = False,
) -> DeclusterScan:
    """Scan cell sizes and choose the one with the lowest (or highest) declustered mean."""
    if n_sizes < 2:
        msg = f'n_sizes must be at least 2, got {n_sizes!r}.'
        raise ValueError(msg)
    dim = 3 if np.ptp(points[:, 2]) > 0 else 2
    if aspect is not None:
        shape = cell_vector(aspect, points)
        dim = shape.size
    else:
        shape = np.ones(dim)
    lo, hi = _size_range(points[:, :dim] / shape, size_range)
    sizes = np.geomspace(lo, hi, n_sizes)
    cells = sizes[:, np.newaxis] * shape
    means = np.array(
        [declustered_mean(values, declustering_weights(points, c, n_offsets)) for c in cells]
    )
    best = int(np.argmax(means) if maximize else np.argmin(means))
    if best in (0, n_sizes - 1):
        end = 'smallest' if best == 0 else 'largest'
        warnings.warn(
            f'The {"highest" if maximize else "lowest"} declustered mean is at the {end} cell '
            f'size tried, so the scan has no interior extremum. Widen size_range, or try '
            f'maximize={not maximize}: the samples may be clustered where the values are '
            f'{"high" if maximize else "low"}.',
            stacklevel=4,
        )
    return DeclusterScan(cell_sizes=cells, means=means, best=best)


def _size_range(coords: np.ndarray, size_range: tuple[float, float] | None) -> tuple[float, float]:
    """Return the smallest and largest cell size to scan.

    By default the smallest is the 10th percentile of the nearest-neighbor
    distances and the largest is half the longest extent. The median would be
    too large, since in clustered samples it is the regular spacing outside the
    clusters.
    """
    if size_range is not None:
        lo, hi = (float(s) for s in size_range)
        if not 0 < lo < hi:
            msg = f'size_range must satisfy 0 < min < max, got {size_range!r}.'
            raise ValueError(msg)
        return lo, hi
    from scipy.spatial import cKDTree  # noqa: PLC0415

    nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
    nearest = nearest[nearest > 0]
    hi = 0.5 * float(np.ptp(coords, axis=0).max())
    lo = float(np.percentile(nearest, 10)) if nearest.size else hi
    if lo >= hi:
        msg = 'Cannot choose a cell size range for these samples. Pass size_range=(min, max).'
        raise ValueError(msg)
    return lo, hi


def decluster(
    dataset: pv.DataSet,
    cell_size: float | tuple[float, ...] | None,
    *,
    scalars: str | None,
    name: str,
    n_offsets: int,
    scan: DeclusterScan | None,
) -> pv.DataSet:
    """Return a copy of ``dataset`` with declustering weights, as point or cell data."""
    points, data = sample_points(dataset, scalars)
    cell = scan.cell_sizes[scan.best] if scan else cell_vector(cell_size, points)  # type: ignore[arg-type]
    result = dataset.copy()
    target = result.cell_data if data == 'cell' else result.point_data
    target[name] = declustering_weights(points, cell, n_offsets)
    result.field_data['declustering cell size'] = cell
    return result


def sample_points(dataset: pv.DataSet, scalars: str | None) -> tuple[np.ndarray, str]:
    """Return sample locations and the association of ``scalars`` (``'point'`` or ``'cell'``).

    Point arrays are sampled at the points and cell arrays at the cell centers.
    Without ``scalars`` the points are used.
    """
    if scalars is None or scalars in dataset.point_data:
        return np.asarray(dataset.points), 'point'
    if scalars in dataset.cell_data:
        return np.asarray(dataset.cell_centers().points), 'cell'
    msg = f'Array {scalars!r} is not in the point or cell data of the samples.'
    raise KeyError(msg)


def scan_dataset(dataset: pv.DataSet, scalars: str | None, **options: Any) -> DeclusterScan:
    """Scan cell sizes for the ``scalars`` array of ``dataset``."""
    if scalars is None:
        msg = 'Choosing a cell size needs scalars, the name of the array whose mean to decluster.'
        raise ValueError(msg)
    points, association = sample_points(dataset, scalars)
    data = dataset.cell_data if association == 'cell' else dataset.point_data
    values = np.asarray(data[scalars], dtype=float)
    return scan_declustering(points, values, **options)
