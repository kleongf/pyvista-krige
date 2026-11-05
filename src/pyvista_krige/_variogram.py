"""Experimental and fitted variograms of a kriging model."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Parameter names of each PyKrige model, in PyKrige's internal order.
_PARAMETER_NAMES = {
    'linear': ('slope', 'nugget'),
    'power': ('scale', 'exponent', 'nugget'),
    'gaussian': ('psill', 'range', 'nugget'),
    'spherical': ('psill', 'range', 'nugget'),
    'exponential': ('psill', 'range', 'nugget'),
    'hole-effect': ('psill', 'range', 'nugget'),
}

# Unit vectors for the named directions of ``experimental_variogram``.
_DIRECTIONS = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


@dataclass(frozen=True)
class Variogram:
    """Experimental semivariance and the variogram model fitted to it.

    Attributes
    ----------
    model_name : str
        PyKrige variogram model name, such as ``'spherical'`` or ``'custom'``.

    parameters : numpy.ndarray
        Fitted model parameters in PyKrige's internal order, for example
        ``[psill, range, nugget]`` for the bounded models.

    lags : numpy.ndarray
        Centers of the lag bins of the experimental variogram.

    semivariance : numpy.ndarray
        Experimental semivariance in each lag bin.

    counts : numpy.ndarray | None
        Number of sample pairs in each lag bin, when known. PyKrige's own fit
        does not report them; ``detrend`` and ``max_lag`` fits do.

    """

    model_name: str
    parameters: np.ndarray
    lags: np.ndarray
    semivariance: np.ndarray
    function: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(repr=False)
    counts: np.ndarray | None = None

    def model(self, distances: np.ndarray) -> np.ndarray:
        """Evaluate the fitted variogram model at ``distances``.

        Parameters
        ----------
        distances : numpy.ndarray
            Separation distances.

        Returns
        -------
        numpy.ndarray
            Model semivariance at each distance.

        """
        return np.asarray(self.function(self.parameters, np.asarray(distances, dtype=float)))

    @property
    def model_values(self) -> np.ndarray:
        """Return the fitted model evaluated at :attr:`lags`."""
        return self.model(self.lags)


@dataclass(frozen=True)
class ExperimentalVariogram:
    """Binned semivariance of sample pairs, optionally along one direction.

    Create one with ``samples.krige.variogram(...)``.

    Attributes
    ----------
    lags : numpy.ndarray
        Mean pair distance in each non-empty lag bin.

    semivariance : numpy.ndarray
        Mean semivariance of the pairs in each bin.

    counts : numpy.ndarray
        Number of pairs in each bin.

    direction : str | tuple[float, ...] | None
        Direction the pairs were selected along, or ``None`` for all
        directions.

    max_lag : float
        Upper edge of the last lag bin.

    """

    lags: np.ndarray
    semivariance: np.ndarray
    counts: np.ndarray
    direction: str | tuple[float, ...] | None
    max_lag: float

    @property
    def label(self) -> str:
        """Return a short description of the direction, for chart legends."""
        if self.direction is None:
            return 'omnidirectional'
        if isinstance(self.direction, str):
            return self.direction
        return '(' + ', '.join(f'{c:g}' for c in self.direction) + ')'

    def fit(self, variogram_model: str) -> Variogram:
        """Fit a PyKrige variogram model to these lags.

        Bins are weighted by their pair counts. The fitted parameters can be
        passed to :meth:`KrigeAccessor.fit` as ``variogram_parameters``, for
        example to compare the horizontal and vertical ranges when choosing
        ``anisotropy_scaling_z``.

        Parameters
        ----------
        variogram_model : str
            ``'linear'``, ``'power'``, ``'gaussian'``, ``'spherical'``,
            ``'exponential'`` or ``'hole-effect'``.

        Returns
        -------
        Variogram
            The fitted model with these lags.

        """
        parameters = fit_model(
            variogram_model, self.lags, self.semivariance, self.counts, self.max_lag
        )
        return Variogram(
            model_name=variogram_model,
            parameters=parameters,
            lags=self.lags,
            semivariance=self.semivariance,
            function=model_function(variogram_model),
            counts=self.counts,
        )


def model_function(variogram_model: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return PyKrige's function for a named variogram model."""
    from pykrige import variogram_models  # noqa: PLC0415

    if variogram_model not in _PARAMETER_NAMES:
        names = ', '.join(repr(name) for name in _PARAMETER_NAMES)
        msg = f'variogram_model must be one of {names} here, got {variogram_model!r}.'
        raise ValueError(msg)
    return getattr(variogram_models, f'{variogram_model.replace("-", "_")}_variogram_model')


def parameter_dict(variogram_model: str, parameters: np.ndarray) -> dict[str, float]:
    """Return ``parameters`` as the dict PyKrige accepts for ``variogram_parameters``."""
    return dict(zip(_PARAMETER_NAMES[variogram_model], map(float, parameters), strict=True))


def fit_model(
    variogram_model: str,
    lags: np.ndarray,
    semivariance: np.ndarray,
    counts: np.ndarray,
    max_lag: float,
) -> np.ndarray:
    """Least-squares fit of a PyKrige model to binned semivariance, weighted by pair counts.

    Starting values and bounds follow PyKrige's own fit.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    function = model_function(variogram_model)
    if len(lags) < 2:
        msg = 'At least two non-empty lag bins are needed to fit a variogram.'
        raise ValueError(msg)
    low, high = float(np.min(semivariance)), float(np.max(semivariance))
    if variogram_model == 'linear':
        x0 = [(high - low) / (lags.max() - lags.min()), low]
        bounds = ([0.0, 0.0], [np.inf, high])
    elif variogram_model == 'power':
        x0 = [(high - low) / (lags.max() - lags.min()), 1.1, low]
        bounds = ([0.0, 0.001, 0.0], [np.inf, 1.999, high])
    else:
        x0 = [high - low, 0.25 * max_lag, low]
        bounds = ([0.0, 0.0, 0.0], [10.0 * high, max_lag, high])
    x0 = np.clip(x0, bounds[0], bounds[1])
    weights = np.sqrt(counts / counts.sum())

    def residuals(params: np.ndarray) -> np.ndarray:
        return weights * (function(params, lags) - semivariance)

    return least_squares(residuals, x0, bounds=bounds, loss='soft_l1').x


def experimental_variogram(
    points: np.ndarray,
    values: np.ndarray,
    *,
    nlags: int,
    max_lag: float | None = None,
    direction: str | tuple[float, ...] | None = None,
    tolerance: float = 22.5,
    bandwidth: float | None = None,
) -> ExperimentalVariogram:
    """Bin the semivariance of every sample pair by distance.

    ``points`` has one column per spatial dimension. Bin ``k`` holds the pairs
    with ``edges[k] < distance <= edges[k + 1]``, and empty bins are dropped.
    ``direction`` keeps only pairs within ``tolerance`` degrees of a direction
    (either sense), and ``bandwidth`` also caps their distance from that line.
    ``'horizontal'`` keeps pairs within ``tolerance`` of the xy plane.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    ndim = points.shape[1]
    if nlags < 1:
        msg = f'nlags must be at least 1, got {nlags}.'
        raise ValueError(msg)
    if max_lag is None:
        max_lag = 0.5 * float(np.linalg.norm(np.ptp(points, axis=0)))
    unit = _direction_vector(direction, ndim)
    cos_tol = np.cos(np.radians(tolerance))
    sin_tol = np.sin(np.radians(tolerance))

    edges = np.linspace(0.0, max_lag, nlags + 1)
    counts = np.zeros(nlags)
    dist_sum = np.zeros(nlags)
    gamma_sum = np.zeros(nlags)
    n = len(values)
    # Process rows in blocks so memory stays near 4M pairs, whatever the sample count.
    block = max(1, 4_000_000 // max(n, 1))
    for start in range(0, n, block):
        rows = np.arange(start, min(start + block, n))
        i, j = np.nonzero(np.arange(n)[np.newaxis, :] > rows[:, np.newaxis])
        i = rows[i]
        delta = points[j] - points[i]
        dist = np.linalg.norm(delta, axis=1)
        keep = (dist > 0) & (dist <= max_lag)
        if direction == 'horizontal':
            keep &= np.abs(delta[:, 2]) <= sin_tol * dist
        elif unit is not None:
            along = np.abs(delta @ unit)
            keep &= along >= cos_tol * dist
            if bandwidth is not None:
                keep &= np.sqrt(np.clip(dist**2 - along**2, 0, None)) <= bandwidth
        dist, i, j = dist[keep], i[keep], j[keep]
        bins = np.clip(np.searchsorted(edges, dist, side='left') - 1, 0, nlags - 1)
        counts += np.bincount(bins, minlength=nlags)
        dist_sum += np.bincount(bins, weights=dist, minlength=nlags)
        gamma_sum += np.bincount(bins, weights=0.5 * (values[i] - values[j]) ** 2, minlength=nlags)

    filled = counts > 0
    if not filled.any():
        msg = 'No sample pairs fall within max_lag and the direction tolerance.'
        raise ValueError(msg)
    if isinstance(direction, (list, np.ndarray)):
        direction = tuple(float(c) for c in direction)
    return ExperimentalVariogram(
        lags=dist_sum[filled] / counts[filled],
        semivariance=gamma_sum[filled] / counts[filled],
        counts=counts[filled],
        direction=direction,
        max_lag=float(max_lag),
    )


def _direction_vector(direction: str | tuple[float, ...] | None, ndim: int) -> np.ndarray | None:
    """Return the unit vector of ``direction``, or ``None`` for all or horizontal pairs."""
    if direction is None or direction == 'horizontal':
        if direction == 'horizontal' and ndim != 3:
            msg = "direction='horizontal' needs 3D samples."
            raise ValueError(msg)
        return None
    if isinstance(direction, str):
        name = 'z' if direction == 'vertical' else direction
        if name not in _DIRECTIONS:
            msg = (
                "direction must be 'x', 'y', 'z', 'vertical', 'horizontal' or a vector, "
                f'got {direction!r}.'
            )
            raise ValueError(msg)
        vector = np.array(_DIRECTIONS[name])
    else:
        vector = np.asarray(direction, dtype=float)
        if vector.size == 2 and ndim == 3:
            vector = np.append(vector, 0.0)
    vector = vector[:ndim] if ndim == 2 and vector.size == 3 else vector
    norm = np.linalg.norm(vector)
    if vector.size != ndim or norm == 0:
        msg = f'direction must be a nonzero {ndim}-vector, got {direction!r}.'
        raise ValueError(msg)
    return vector / norm
