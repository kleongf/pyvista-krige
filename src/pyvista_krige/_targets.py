"""Translate between PyVista datasets and PyKrige's array interface."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pyvista as pv


def grid_axes(target: pv.DataSet, dim: int) -> tuple[np.ndarray, ...] | None:
    """Return the axis vectors of an axis-aligned structured ``target``, else ``None``.

    PyKrige's ``'grid'`` style returns arrays shaped ``(ny, nx)`` or
    ``(nz, ny, nx)`` in C order, so raveling them gives x-fastest point
    order, which is VTK's structured point order.
    """
    if not isinstance(target, (pv.ImageData, pv.RectilinearGrid)):
        return None
    nx, ny, nz = target.dimensions
    points = target.points.reshape(nz, ny, nx, 3)
    x = points[0, 0, :, 0]
    y = points[0, :, 0, 1]
    z = points[:, 0, 0, 2]
    aligned = (
        np.allclose(points[..., 0], x[np.newaxis, np.newaxis, :])
        and np.allclose(points[..., 1], y[np.newaxis, :, np.newaxis])
        and np.allclose(points[..., 2], z[:, np.newaxis, np.newaxis])
    )
    if not aligned:
        return None
    if dim == 2:
        return (x, y) if nz == 1 else None
    return x, y, z


class Samples(NamedTuple):
    """Finite samples of one array, plus how to align other arrays with them."""

    points: np.ndarray
    values: np.ndarray
    data: pv.DataSetAttributes
    keep: np.ndarray

    def aligned(self, name: str) -> np.ndarray:
        """Return array ``name`` with the same association and filtering as the values."""
        return np.asarray(self.data[name], dtype=float)[self.keep]


def sample_arrays(samples: pv.DataSet, scalars: str) -> Samples:
    """Return the finite sample locations and values for the ``scalars`` array.

    Point arrays are sampled at the points; cell arrays at the cell centers.
    """
    if scalars in samples.point_data:
        points, data = samples.points, samples.point_data
    elif scalars in samples.cell_data:
        points, data = samples.cell_centers().points, samples.cell_data
    else:
        msg = f'Array {scalars!r} is not in the point or cell data of the samples.'
        raise KeyError(msg)
    values = np.asarray(data[scalars], dtype=float)
    keep = np.isfinite(values)
    if not keep.all():
        n_dropped = int((~keep).sum())
        warnings.warn(f'Dropping {n_dropped} samples with NaN or infinite values.', stacklevel=4)
    return Samples(points[keep], values[keep], data, keep)


def cell_centers(target: pv.DataSet) -> pv.DataSet:
    """Return a dataset with one point at the center of each cell of ``target``, in cell order.

    The centers of an axis-aligned ``ImageData`` or ``RectilinearGrid`` form a
    ``RectilinearGrid``, so kriging them keeps PyKrige's fast grid path. VTK
    numbers structured cells x-fastest, like structured points.
    """
    axes = grid_axes(target, 3)
    if axes is None:
        return pv.PolyData(target.cell_centers().points)
    # An axis with one point is flat: its cells sit on that coordinate.
    return pv.RectilinearGrid(*(0.5 * (a[:-1] + a[1:]) if a.size > 1 else a for a in axes))


def resolve_where(
    target: pv.DataSet, where: str | np.ndarray | None, preference: str = 'point'
) -> np.ndarray | None:
    """Return ``where`` as a boolean array over the points or cells of ``target``, or ``None``."""
    if where is None:
        return None
    if preference == 'cell':
        data, size = target.cell_data, target.n_cells
    else:
        data, size = target.point_data, target.n_points
    mask = data[where] if isinstance(where, str) else where
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != size:
        msg = f'where must have one value per target {preference} ({size}), got {mask.size}.'
        raise ValueError(msg)
    return mask


def execute_on(
    instance: object,
    target: pv.DataSet,
    dim: int,
    where: np.ndarray | None,
    drift_arrays: list[np.ndarray] | None = None,
    max_points: int | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Krige onto the points of ``target`` and return flat estimate and variance arrays.

    ``drift_arrays`` hold one value per target point and are reshaped to match the
    PyKrige style used. Points excluded by ``where`` hold unspecified values; the
    caller overwrites them. At most ``max_points`` points are kriged per PyKrige
    call: grids are split into slabs along their slowest axis, and other targets
    into runs of points.
    """
    axes = grid_axes(target, dim)
    if axes is not None:
        shape = tuple(len(axis) for axis in reversed(axes))
        per_slab = int(np.prod(shape[1:]))
        if max_points is None or target.n_points <= max_points:
            rows = shape[0]
        else:
            rows = max_points // per_slab
        if rows >= 1:
            return _execute_grid(instance, axes, shape, rows, where, drift_arrays, kwargs)

    estimate = np.full(target.n_points, np.nan)
    sigma2 = np.full(target.n_points, np.nan)
    selected = np.flatnonzero(np.ones(target.n_points, dtype=bool) if where is None else where)
    step = len(selected) if max_points is None else max_points
    for start in range(0, len(selected), max(step, 1)):
        chunk = selected[start : start + step]
        if drift_arrays is not None:
            kwargs['specified_drift_arrays'] = [a[chunk] for a in drift_arrays]
        est, var = instance.execute('points', *target.points[chunk].T[:dim], **kwargs)
        estimate[chunk] = np.ma.getdata(est)
        sigma2[chunk] = np.ma.getdata(var)
    return estimate, sigma2


def _execute_grid(
    instance: object,
    axes: tuple[np.ndarray, ...],
    shape: tuple[int, ...],
    rows: int,
    where: np.ndarray | None,
    drift_arrays: list[np.ndarray] | None,
    kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Krige a grid in slabs of ``rows`` along its slowest axis (z in 3D, y in 2D)."""
    # PyKrige's mask is True where a node is skipped, shaped like the grid.
    skip = None if where is None else ~where.reshape(shape)
    drifts = None if drift_arrays is None else [a.reshape(shape) for a in drift_arrays]
    estimates, variances = [], []
    for start in range(0, shape[0], rows):
        part = slice(start, start + rows)
        slab_axes = (*axes[:-1], axes[-1][part])
        if drifts is not None:
            kwargs['specified_drift_arrays'] = [a[part] for a in drifts]
        if skip is None:
            est, var = instance.execute('grid', *slab_axes, **kwargs)
        elif skip[part].all():
            est = var = np.full(skip[part].shape, np.nan)
        else:
            est, var = instance.execute('masked', *slab_axes, mask=skip[part], **kwargs)
        estimates.append(np.ma.getdata(est))
        variances.append(np.ma.getdata(var))
    return np.concatenate(estimates).ravel(), np.concatenate(variances).ravel()


def near_samples(
    sample_points: np.ndarray, target: pv.DataSet, dim: int, max_distance: float
) -> np.ndarray:
    """Return whether each target point lies within ``max_distance`` of a sample."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    distance, _ = cKDTree(sample_points[:, :dim]).query(target.points[:, :dim])
    return distance <= max_distance


def image_spanning(
    bounds: tuple[float, ...],
    dim: int,
    *,
    dimensions: tuple[int, ...] | None,
    spacing: float | tuple[float, ...] | None,
    padding: float,
) -> pv.ImageData:
    """Return an ``ImageData`` whose points span ``bounds`` grown by ``padding``.

    ``padding`` is a fraction of the extent on each side. With ``dimensions`` the
    outer points lie on the padded bounds; with ``spacing`` the image covers them.
    A 2D model gets a single layer at the lower z bound.
    """
    lower = np.array(bounds[::2], dtype=float)
    upper = np.array(bounds[1::2], dtype=float)
    pad = padding * (upper - lower)
    lower, upper = lower - pad, upper + pad
    if dim == 2:
        upper[2] = lower[2]
    extent = upper - lower
    if dimensions is not None and spacing is not None:
        msg = 'Pass either dimensions or spacing, not both.'
        raise ValueError(msg)
    if spacing is not None:
        step = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
        dims = np.ceil(extent / step - 1e-9).astype(int) + 1
    else:
        dims = np.ones(3, dtype=int)
        dims[:dim] = 50 if dimensions is None else dimensions[:dim]
        step = np.divide(extent, dims - 1, out=np.ones(3), where=dims > 1)
    if dim == 2:
        dims[2], step[2] = 1, 1.0
    return pv.ImageData(dimensions=tuple(dims), spacing=tuple(step), origin=tuple(lower))
