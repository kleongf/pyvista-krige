"""Transforms that let skewed data be kriged in a roughly Gaussian space."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import ndtr
from scipy.special import ndtri

# Trapezoid rule for the expectation over a standard normal variable. The
# normal score back-transform has a kink at every sample, which Gauss-Hermite
# rules integrate poorly; evenly spaced nodes keep the error near 1 %.
_NODES = np.linspace(-6.0, 6.0, 81)
_WEIGHTS = np.exp(-(_NODES**2) / 2)
_WEIGHTS /= _WEIGHTS.sum()

# Normal scores have unit variance, so smaller kriging variances are round-off
# of the kriging solve. Left in, they would shift the mean at a sample by about
# their standard deviation, through the kink in the table there.
_ROUND_OFF = 1e-8


class LogTransform:
    """Natural logarithm, for positive, lognormally distributed data.

    :meth:`back` uses the lognormal mean and variance. For ordinary kriging the
    exact back-transform also subtracts the kriging system's Lagrange
    multiplier, which PyKrige does not expose. Without it the estimate is
    biased slightly high where the kriging variance exceeds the simple kriging
    variance, far from the samples.
    """

    name: ClassVar[str] = 'log'

    @classmethod
    def from_samples(
        cls, values: np.ndarray, limits: tuple[float | None, float | None] | None = None
    ) -> LogTransform:
        """Check that ``values`` can be log-transformed and return the transform."""
        if limits is not None:
            msg = "transform_limits applies only to transform='normal_score'."
            raise ValueError(msg)
        if np.any(values <= 0):
            msg = f"transform='log' needs positive values, but the smallest is {np.min(values):g}."
            raise ValueError(msg)
        return cls()

    def forward(self, values: np.ndarray | float) -> np.ndarray:
        """Return the logarithm of ``values``, with ``-inf`` for values of zero or less."""
        values = np.asarray(values, dtype=float)
        with np.errstate(divide='ignore'):
            return np.log(np.where(values > 0, values, 0.0))

    def back(self, estimate: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and variance of the lognormal with these log-space moments."""
        variance = np.clip(variance, 0, None)
        with np.errstate(over='ignore', invalid='ignore'):
            mean = np.exp(estimate + variance / 2)
            return mean, mean**2 * np.expm1(variance)


class NormalScoreTransform:
    """Map values to standard normal scores through their empirical distribution.

    The cumulative distribution is piecewise linear through the samples, at
    plotting positions ``(rank - 0.5) / n``, and extends linearly to
    probability 0 at the lower limit and 1 at the upper limit. Values outside
    the limits map to infinite scores.

    Parameters
    ----------
    values : numpy.ndarray
        Strictly increasing table of values, starting at the lower limit and
        ending at the upper limit.

    cdf : numpy.ndarray
        Cumulative probability of each value, from 0 to 1.

    """

    name: ClassVar[str] = 'normal_score'

    def __init__(self, values: np.ndarray, cdf: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float)
        self.cdf = np.asarray(cdf, dtype=float)

    @classmethod
    def from_samples(
        cls, values: np.ndarray, limits: tuple[float | None, float | None] | None = None
    ) -> NormalScoreTransform:
        """Build the table from sample ``values``.

        ``limits`` gives the values where the distribution reaches probability
        0 and 1; ``None`` for either extends the outermost segment of the
        empirical distribution until it does.
        """
        unique, counts = np.unique(values, return_counts=True)
        if len(unique) < 2:
            msg = "transform='normal_score' needs at least two distinct values."
            raise ValueError(msg)
        # Tied samples share the mean plotting position of their ranks.
        cdf = (np.cumsum(counts) - counts / 2) / len(values)
        low = unique[0] - (unique[1] - unique[0]) * cdf[0] / (cdf[1] - cdf[0])
        high = unique[-1] + (unique[-1] - unique[-2]) * (1 - cdf[-1]) / (cdf[-1] - cdf[-2])
        if limits is not None:
            low = low if limits[0] is None else float(limits[0])
            high = high if limits[1] is None else float(limits[1])
        if not low < unique[0] or not high > unique[-1]:
            msg = (
                f'transform_limits must lie outside the sample range [{unique[0]:g}, '
                f'{unique[-1]:g}], got ({low:g}, {high:g}).'
            )
            raise ValueError(msg)
        return cls(np.concatenate([[low], unique, [high]]), np.concatenate([[0.0], cdf, [1.0]]))

    def forward(self, values: np.ndarray | float) -> np.ndarray:
        """Return the normal scores of ``values``."""
        return ndtri(np.interp(values, self.values, self.cdf))

    def back(self, estimate: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and variance of the values whose scores are normal with these moments.

        Integrates over the normal distribution of the score at each point by
        quadrature. A zero variance gives the inverse transform.
        """
        std = np.sqrt(np.where(variance > _ROUND_OFF, variance, 0.0))
        mean = np.zeros_like(estimate, dtype=float)
        square = np.zeros_like(mean)
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            values = np.interp(ndtr(estimate + std * node), self.cdf, self.values)
            mean += weight * values
            square += weight * values**2
        return mean, np.clip(square - mean**2, 0, None)


Transform = LogTransform | NormalScoreTransform

TRANSFORMS: dict[str, type[Transform]] = {
    LogTransform.name: LogTransform,
    NormalScoreTransform.name: NormalScoreTransform,
}


def fit_transform(
    transform: str | None,
    values: np.ndarray,
    limits: tuple[float | None, float | None] | None,
) -> Transform | None:
    """Return the transform named ``transform``, fitted to the sample ``values``."""
    if transform is None:
        if limits is not None:
            msg = "transform_limits applies only to transform='normal_score'."
            raise ValueError(msg)
        return None
    if transform not in TRANSFORMS:
        msg = f"transform must be None, 'log' or 'normal_score', got {transform!r}."
        raise ValueError(msg)
    return TRANSFORMS[transform].from_samples(values, limits)
