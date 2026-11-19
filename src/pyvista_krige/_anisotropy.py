"""Estimate geometric anisotropy from directional variograms."""

from __future__ import annotations

import numpy as np

from pyvista_krige._variogram import ExperimentalVariogram
from pyvista_krige._variogram import experimental_variogram
from pyvista_krige._variogram import fit_model
from pyvista_krige._variogram import model_function

# Models with a finite range; anisotropy compares the ranges of different directions.
BOUNDED_MODELS = ('gaussian', 'spherical', 'exponential', 'hole-effect')

_AZIMUTHS = np.arange(0.0, 180.0, 22.5)
_TOLERANCE = 22.5
_MIN_DIRECTIONS = 3


def estimate_anisotropy(
    coords: np.ndarray,
    values: np.ndarray,
    variogram_model: str,
    *,
    nlags: int,
    max_lag: float | None = None,
) -> dict[str, float]:
    """Estimate PyKrige's anisotropy options from the ranges of directional variograms.

    The nugget and sill are fitted once to the omnidirectional variogram, then
    only the range is fitted along eight horizontal azimuths. An ellipse fitted
    to those ranges gives the major axis (``anisotropy_angle``, degrees
    counter-clockwise from +x) and the ratio of its axes. For 3D samples the
    vertical range gives ``anisotropy_scaling_z``. Tilted anisotropy is not
    estimated, so the x and y rotation angles stay at PyKrige's defaults.

    Returns ``anisotropy_scaling`` and ``anisotropy_angle`` for 2D ``coords``,
    or ``anisotropy_scaling_y``, ``anisotropy_scaling_z`` and
    ``anisotropy_angle_z`` for 3D.
    """
    if variogram_model not in BOUNDED_MODELS:
        names = ', '.join(repr(name) for name in BOUNDED_MODELS)
        msg = (
            "anisotropy='auto' compares variogram ranges, so it needs a bounded model "
            f"({names}), got {variogram_model!r}. PyKrige defaults to 'linear'; pass, "
            "for example, variogram_model='spherical'."
        )
        raise ValueError(msg)
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    ndim = coords.shape[1]
    if max_lag is None:
        max_lag = 0.5 * float(np.linalg.norm(np.ptp(coords, axis=0)))

    omni = experimental_variogram(coords, values, nlags=nlags, max_lag=max_lag)
    psill, _, nugget = fit_model(
        variogram_model, omni.lags, omni.semivariance, omni.counts, omni.max_lag
    )

    directions, ranges, weights = [], [], []
    for azimuth in np.radians(_AZIMUTHS):
        unit = np.array([np.cos(azimuth), np.sin(azimuth)])
        vector = unit if ndim == 2 else (*unit, 0.0)
        fitted = _directional_range(
            coords, values, vector, variogram_model, psill, nugget, nlags, max_lag
        )
        if fitted is not None:
            directions.append(unit)
            ranges.append(fitted[0])
            weights.append(np.sqrt(fitted[1]))
    if len(ranges) < _MIN_DIRECTIONS:
        msg = (
            f'Too few sample pairs per direction to estimate anisotropy: only {len(ranges)} '
            f'of {len(_AZIMUTHS)} horizontal directions have two or more non-empty lag bins. '
            'Use more samples, a larger max_lag, or set the anisotropy options manually.'
        )
        raise ValueError(msg)

    major, minor, angle = _fit_ellipse(
        np.array(directions), np.array(ranges), np.array(weights), max_lag
    )
    if ndim == 2:
        return {'anisotropy_scaling': major / minor, 'anisotropy_angle': angle}

    vertical = _directional_range(
        coords, values, (0.0, 0.0, 1.0), variogram_model, psill, nugget, nlags, max_lag
    )
    if vertical is None:
        msg = (
            'Too few vertical sample pairs to estimate anisotropy. Use more samples, a larger '
            'max_lag, or set anisotropy_scaling_z manually.'
        )
        raise ValueError(msg)
    return {
        'anisotropy_scaling_y': major / minor,
        'anisotropy_scaling_z': major / vertical[0],
        'anisotropy_angle_z': angle,
    }


def _directional_range(
    coords: np.ndarray,
    values: np.ndarray,
    direction: tuple[float, ...] | np.ndarray,
    variogram_model: str,
    psill: float,
    nugget: float,
    nlags: int,
    max_lag: float,
) -> tuple[float, float] | None:
    """Fit the range along ``direction`` with the sill fixed; return it and the pair count.

    Returns ``None`` when the direction has too few pairs to fit.
    """
    try:
        vario = experimental_variogram(
            coords,
            values,
            nlags=nlags,
            max_lag=max_lag,
            direction=tuple(float(c) for c in direction),
            tolerance=_TOLERANCE,
        )
    except ValueError:
        return None
    if len(vario.lags) < 2:
        return None
    return _fit_range(vario, variogram_model, psill, nugget), float(vario.counts.sum())


def _fit_range(
    vario: ExperimentalVariogram, variogram_model: str, psill: float, nugget: float
) -> float:
    """Least-squares range of a bounded model with the sill and nugget held fixed."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    function = model_function(variogram_model)
    weights = np.sqrt(vario.counts / vario.counts.sum())
    # A range below the smallest lag cannot be told from a pure nugget.
    low, high = float(vario.lags.min()), vario.max_lag

    def residuals(range_: np.ndarray) -> np.ndarray:
        parameters = np.array([psill, range_[0], nugget])
        return weights * (function(parameters, vario.lags) - vario.semivariance)

    # The misfit can have several minima in the range, so start from the best of a coarse scan.
    scan = np.geomspace(low, high, 40)
    start = scan[np.argmin([np.sum(residuals(np.array([r])) ** 2) for r in scan])]
    return float(least_squares(residuals, [start], bounds=([low], [high]), loss='soft_l1').x[0])


def _fit_ellipse(
    directions: np.ndarray, ranges: np.ndarray, weights: np.ndarray, max_lag: float
) -> tuple[float, float, float]:
    """Fit ``u.T @ M @ u = 1 / range**2`` and return the major range, minor range and angle."""
    c, s = directions.T
    design = np.column_stack([c**2, 2 * c * s, s**2])
    # Scaling each row by range**2 weighs the relative error in 1 / range**2; without it the
    # short ranges dominate and the long axis is fitted poorly.
    weights = weights * ranges**2
    solution = np.linalg.lstsq(design * weights[:, None], weights / ranges**2, rcond=None)[0]
    matrix = np.array([[solution[0], solution[1]], [solution[1], solution[2]]])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Ranges cannot exceed max_lag, which also keeps the matrix positive definite.
    eigenvalues = np.clip(eigenvalues, 1.0 / max_lag**2, None)
    major_axis = eigenvectors[:, 0]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180.0)
    return float(1.0 / np.sqrt(eigenvalues[0])), float(1.0 / np.sqrt(eigenvalues[1])), angle
