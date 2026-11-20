"""Sequential Gaussian simulation conditioned on the samples of a kriging model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import ndtr

from pyvista_krige._transform import NormalScoreTransform

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista_krige._model import KrigingModel

# Target points this close to a conditioning point, relative to the extent of
# all points, take its value instead of being kriged from a singular system.
_SAME_POINT = 1e-9


def simulate(
    model: KrigingModel,
    target: pv.DataSet,
    n: int,
    *,
    seed: int | np.random.Generator | None,
    n_neighbors: int,
    summary: bool,
    exceed: float | None,
) -> pv.DataSet:
    """Return a copy of ``target`` with ``n`` realizations or their summaries.

    See :meth:`KrigingModel.simulate`, which is the usual way to call this.
    """
    if isinstance(target, pv.MultiBlock):
        msg = 'MultiBlock targets are not supported. Use target.combine() first.'
        raise TypeError(msg)
    if model._method != 'ordinary':
        msg = 'simulate supports ordinary kriging models only.'
        raise ValueError(msg)
    instance = model.instance
    if getattr(instance, 'coordinates_type', 'euclidean') != 'euclidean':
        msg = "simulate does not support coordinates_type='geographic'."
        raise ValueError(msg)
    if n < 1:
        msg = f'n must be at least 1, got {n}.'
        raise ValueError(msg)
    if n_neighbors < 1:
        msg = f'n_neighbors must be at least 1, got {n_neighbors}.'
        raise ValueError(msg)

    scalars = model._scalars
    values = np.asarray(model.samples[scalars], dtype=float)
    transform = model.transform
    if isinstance(transform, NormalScoreTransform):
        variogram_scale = 1.0
    else:
        # The variogram was fitted to the kriged values; rescale it to unit
        # variance, the variance of the normal scores it is applied to.
        kriged = values if transform is None else transform.forward(values)
        variogram_scale = float(np.var(kriged)) or 1.0
        transform = NormalScoreTransform.from_samples(values)

    def variogram(distance: np.ndarray) -> np.ndarray:
        gamma = instance.variogram_function(instance.variogram_model_parameters, distance)
        return np.asarray(gamma, dtype=float) / variogram_scale

    sample_coords = _sample_coords(instance, model.dim)
    node_coords = _adjust(instance, model.dim, np.asarray(target.points, dtype=float))
    scores = simulate_scores(
        sample_coords,
        transform.forward(values),
        node_coords,
        n,
        variogram,
        np.random.default_rng(seed),
        n_neighbors,
    )
    realizations = np.interp(ndtr(scores), transform.cdf, transform.values)
    # Copy the data exactly where a target point is a sample, not its round trip.
    gap, nearest = cKDTree(sample_coords).query(node_coords)
    at_sample = gap <= _SAME_POINT * _extent(sample_coords, node_coords)
    realizations[at_sample] = values[nearest[at_sample], None]

    out = target.copy()
    if not summary and exceed is None:
        for i in range(n):
            out.point_data[f'{scalars} realization {i}'] = realizations[:, i]
        return out
    if summary:
        out.point_data[f'{scalars} mean'] = realizations.mean(axis=1)
        p10, p50, p90 = np.percentile(realizations, [10, 50, 90], axis=1)
        out.point_data[f'{scalars} P10'] = p10
        out.point_data[f'{scalars} P50'] = p50
        out.point_data[f'{scalars} P90'] = p90
    if exceed is not None:
        out.point_data[f'P({scalars} > {exceed:g})'] = (realizations > exceed).mean(axis=1)
    return out


def simulate_scores(
    sample_coords: np.ndarray,
    sample_scores: np.ndarray,
    node_coords: np.ndarray,
    n: int,
    variogram: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_neighbors: int,
) -> np.ndarray:
    """Return ``n`` Gaussian realizations at ``node_coords``, shape ``(nodes, n)``.

    Visits the nodes in one random path shared by every realization, so each
    node's ordinary kriging weights are solved once. Each node is kriged from
    its ``n_neighbors`` nearest samples and earlier nodes, and draws from the
    normal distribution of that estimate and variance. A node at the same place
    as a sample or earlier node takes its value and is not used again, since a
    second point at one place would make the kriging system singular.
    """
    m, total = len(sample_coords), len(node_coords)
    known = np.empty((m + total, n))
    known[:m] = sample_scores[:, None]
    coords = np.vstack([sample_coords, node_coords])
    sample_tree, node_tree = cKDTree(sample_coords), cKDTree(node_coords)
    same = _SAME_POINT * _extent(sample_coords, node_coords)
    path = rng.permutation(total)
    rank = np.empty(total, dtype=int)
    rank[path] = np.arange(total)
    kriged = 0  # visited nodes that condition later ones
    for step, node in enumerate(path):
        point = node_coords[node]
        gap, rows = _nearest(sample_tree, point, min(n_neighbors, m))
        if kriged:
            node_gap, nodes = _nearest_visited(node_tree, rank, point, step, n_neighbors, kriged)
            gap = np.concatenate([gap, node_gap])
            rows = np.concatenate([rows, m + nodes])
            order = np.argsort(gap, kind='stable')[:n_neighbors]
            gap, rows = gap[order], rows[order]
        if gap[0] <= same:
            known[m + node] = known[rows[0]]
            rank[node] = total
            continue
        kriged += 1
        weights, variance = _ordinary_weights(coords[rows], gap, variogram)
        std = np.sqrt(max(variance, 0.0))
        known[m + node] = weights @ known[rows] + std * rng.standard_normal(n)
    return known[m:]


def _ordinary_weights(
    neighbors: np.ndarray,
    gap: np.ndarray,
    variogram: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, float]:
    """Return the ordinary kriging weights of ``neighbors`` and the kriging variance."""
    k = len(neighbors)
    system = np.ones((k + 1, k + 1))
    system[:k, :k] = variogram(cdist(neighbors, neighbors))
    np.fill_diagonal(system[:k, :k], 0.0)
    system[k, k] = 0.0
    right = np.ones(k + 1)
    right[:k] = variogram(gap)
    try:
        solution = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, right)[0]
    return solution[:k], float(solution @ right)


def _nearest(tree: cKDTree, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the distances and indices of the ``k`` points of ``tree`` nearest ``point``."""
    gap, index = tree.query(point, k)
    return np.atleast_1d(gap), np.atleast_1d(index)


def _nearest_visited(
    tree: cKDTree, rank: np.ndarray, point: np.ndarray, step: int, k: int, kriged: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` nodes nearest ``point`` among those ranked before ``step``.

    The ``kriged`` nodes ranked before ``step`` are a random subset of the
    nodes, so asking for about twice ``k`` over their fraction usually finds
    ``k``; otherwise the query doubles until it does or covers every node.
    """
    total = len(rank)
    want = min(k, kriged)
    query = min(total, max(2 * k, -(-2 * k * total // kriged)))
    while True:
        gap, index = _nearest(tree, point, query)
        visited = rank[index] < step
        if visited.sum() >= want or query == total:
            return gap[visited][:want], index[visited][:want]
        query = min(total, 2 * query)


def _sample_coords(instance: object, dim: int) -> np.ndarray:
    """Return PyKrige's anisotropy-adjusted sample coordinates."""
    axes = [instance.X_ADJUSTED, instance.Y_ADJUSTED]
    if dim == 3:
        axes.append(instance.Z_ADJUSTED)
    return np.column_stack(axes).astype(float)


def _adjust(instance: object, dim: int, points: np.ndarray) -> np.ndarray:
    """Map ``points`` into PyKrige's anisotropy-adjusted coordinates, as ``execute`` does."""
    from pykrige.core import _adjust_for_anisotropy  # noqa: PLC0415

    if dim == 2:
        return _adjust_for_anisotropy(
            points[:, :2].copy(),
            [instance.XCENTER, instance.YCENTER],
            [instance.anisotropy_scaling],
            [instance.anisotropy_angle],
        )
    return _adjust_for_anisotropy(
        points[:, :3].copy(),
        [instance.XCENTER, instance.YCENTER, instance.ZCENTER],
        [instance.anisotropy_scaling_y, instance.anisotropy_scaling_z],
        [instance.anisotropy_angle_x, instance.anisotropy_angle_y, instance.anisotropy_angle_z],
    )


def _extent(*coords: np.ndarray) -> float:
    """Return the largest coordinate range of all ``coords`` together, or 1 if it is 0."""
    stacked = np.vstack(coords)
    return float(np.ptp(stacked, axis=0).max()) or 1.0
