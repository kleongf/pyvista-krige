"""Experimental and fitted variogram of a kriging model."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


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

    """

    model_name: str
    parameters: np.ndarray
    lags: np.ndarray
    semivariance: np.ndarray
    function: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(repr=False)

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
