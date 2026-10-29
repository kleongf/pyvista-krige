"""Fitted kriging model that predicts onto PyVista datasets."""

from __future__ import annotations

import copy
from typing import Any
from typing import NamedTuple

import numpy as np
import pyvista as pv

from pyvista_krige._advanced import resolve_dim
from pyvista_krige._advanced import translate_drift
from pyvista_krige._targets import execute_on
from pyvista_krige._targets import resolve_where
from pyvista_krige._targets import sample_arrays
from pyvista_krige._variogram import Variogram


class Statistics(NamedTuple):
    """Cross-validation statistics of a kriging fit, as computed by PyKrige.

    Attributes
    ----------
    q1 : float
        Absolute mean of the standardized residuals; ideally close to 0.

    q2 : float
        Mean of the squared standardized residuals; ideally close to 1.

    cr : float
        ``q2`` times the geometric mean of the kriging variances; smaller is
        better when comparing variogram models.

    residuals : numpy.ndarray
        PyKrige's sequential epsilon residuals, one per sample after the first.

    """

    q1: float
    q2: float
    cr: float
    residuals: np.ndarray


class KrigingModel:
    """A kriging model fitted to scattered samples.

    Create one with ``samples.krige.fit(...)`` rather than directly.

    """

    def __init__(
        self,
        instance: Any,
        scalars: str,
        *,
        method: str,
        dim: int,
        n_samples: int,
        drift_names: list[str] | None = None,
    ) -> None:
        self._instance = instance
        self._scalars = scalars
        self._method = method
        self._dim = dim
        self._n_samples = n_samples
        self._drift_names = drift_names

    @classmethod
    def fit(
        cls,
        samples: pv.DataSet,
        scalars: str,
        *,
        method: str = 'ordinary',
        dim: int | None = None,
        **kwargs: Any,
    ) -> KrigingModel:
        """Fit a model to the ``scalars`` array of ``samples``.

        See :meth:`KrigeAccessor.fit`, which is the usual way to call this.

        Parameters
        ----------
        samples : pyvista.DataSet
            Dataset holding the samples.

        scalars : str
            Name of the point or cell data array holding the sampled values.

        method : str, default: 'ordinary'
            ``'ordinary'`` or ``'universal'`` kriging.

        dim : int, optional
            Spatial dimension, 2 or 3.

        **kwargs : dict, optional
            Options forwarded to the PyKrige class.

        Returns
        -------
        KrigingModel
            The fitted model.

        """
        found = sample_arrays(samples, scalars)
        points, values = found.points, found.values
        drift_names = translate_drift(found, kwargs)
        if method not in ('ordinary', 'universal'):
            msg = f"method must be 'ordinary' or 'universal', got {method!r}."
            raise ValueError(msg)
        dim = resolve_dim(points, kwargs, dim, method)
        kriging_class = _kriging_class(method, dim)
        instance = kriging_class(*points.T[:dim], values, **kwargs)
        return cls(
            instance,
            scalars,
            method=method,
            dim=dim,
            n_samples=len(values),
            drift_names=drift_names,
        )

    @property
    def dim(self) -> int:
        """Return the spatial dimension of the model, 2 or 3."""
        return self._dim

    @property
    def variogram(self) -> Variogram:
        """Return the experimental variogram and the fitted model."""
        instance = self._instance
        return Variogram(
            model_name=instance.variogram_model,
            parameters=np.asarray(instance.variogram_model_parameters, dtype=float),
            lags=np.asarray(instance.lags, dtype=float),
            semivariance=np.asarray(instance.semivariance, dtype=float),
            function=instance.variogram_function,
        )

    @property
    def statistics(self) -> Statistics:
        """Return PyKrige's cross-validation statistics of the fit.

        Raises
        ------
        ValueError
            If 2D ordinary kriging was fitted without ``enable_statistics=True``.

        """
        q1, q2, cr = self._instance.get_statistics()
        if q1 is None:
            msg = 'Statistics are not available. Fit with enable_statistics=True.'
            raise ValueError(msg)
        residuals = np.asarray(self._instance.get_epsilon_residuals(), dtype=float)
        return Statistics(float(q1), float(q2), float(cr), residuals)

    def with_variogram(self, variogram_model: Any, **kwargs: Any) -> KrigingModel:
        """Return a copy of this model refitted with another variogram.

        Parameters
        ----------
        variogram_model : str | gstools.CovModel
            Variogram model, as accepted by :meth:`KrigeAccessor.fit`.

        **kwargs : dict, optional
            Options forwarded to PyKrige's ``update_variogram_model``, such as
            ``variogram_parameters`` or ``nlags``.

        Returns
        -------
        KrigingModel
            The refitted model. This model is unchanged.

        """
        instance = copy.deepcopy(self._instance)
        instance.update_variogram_model(variogram_model, **kwargs)
        return KrigingModel(
            instance,
            self._scalars,
            method=self._method,
            dim=self._dim,
            n_samples=self._n_samples,
            drift_names=self._drift_names,
        )

    @property
    def instance(self) -> Any:
        """Return the underlying PyKrige object."""
        return self._instance

    def predict(
        self,
        target: pv.DataSet,
        *,
        backend: str | None = None,
        n_closest_points: int | None = None,
        max_memory: float | None = 1e9,
        variance: str | bool = True,
        std: bool = False,
        where: str | np.ndarray | None = None,
        null_value: float = np.nan,
    ) -> pv.DataSet:
        """Krige onto the points of ``target`` and return a copy with the estimate.

        Axis-aligned :class:`pyvista.ImageData` and :class:`pyvista.RectilinearGrid`
        targets use PyKrige's fast grid path; every other dataset is kriged
        point by point.

        Parameters
        ----------
        target : pyvista.DataSet
            Dataset whose points are kriged. It is not modified.

        backend : str, optional
            PyKrige backend: ``'vectorized'`` (default), ``'loop'``, or ``'C'``
            (2D ordinary kriging only).

        n_closest_points : int, optional
            Krige each point from only its nearest samples (ordinary kriging
            only). Uses the ``'loop'`` backend unless ``backend='C'``.

        max_memory : float | None, default: 1e9
            Without ``n_closest_points`` PyKrige builds a dense distance matrix
            between targets and samples. Raise before it would exceed this many
            bytes. ``None`` disables the check.

        variance : str | bool, default: True
            Write the kriging variance as ``'<scalars> variance'``, or under
            this name. ``False`` skips it.

        std : bool, default: False
            Write the standard deviation (``'<scalars> std'``) instead of the
            variance.

        where : str | numpy.ndarray, optional
            Boolean array, or name of a target point array, that is ``True``
            where to krige. Other points get ``null_value``, and a
            ``vtkValidPointMask`` array marks the kriged points.

        null_value : float, default: nan
            Value written to points excluded by ``where``.

        Returns
        -------
        pyvista.DataSet
            Copy of ``target`` with the estimate in ``point_data[scalars]``.

        Raises
        ------
        TypeError
            If ``target`` is a :class:`pyvista.MultiBlock`.

        ValueError
            If the backend options are unsupported for this model, or the
            distance matrix would exceed ``max_memory``.

        """
        if isinstance(target, pv.MultiBlock):
            msg = 'MultiBlock targets are not supported. Use target.combine() first.'
            raise TypeError(msg)
        execute_kwargs = self._execute_kwargs(backend, n_closest_points)
        where = resolve_where(target, where)
        n_krige = target.n_points if where is None else int(where.sum())
        if n_closest_points is None and max_memory is not None:
            self._check_memory(n_krige, max_memory)
        drift_arrays = None
        if self._drift_names is not None:
            drift_arrays = [
                np.asarray(target.point_data[name], dtype=float) for name in self._drift_names
            ]
        estimate, sigma2 = execute_on(
            self._instance, target, self._dim, where, drift_arrays, **execute_kwargs
        )
        if std:
            # Round-off can leave tiny negative variances at the samples.
            sigma2 = np.sqrt(np.clip(sigma2, 0, None))
        out = target.copy()
        arrays = {self._scalars: estimate}
        if variance:
            suffix = 'std' if std else 'variance'
            arrays[variance if isinstance(variance, str) else f'{self._scalars} {suffix}'] = sigma2
        for name, values in arrays.items():
            if where is not None:
                values[~where] = null_value
            out.point_data[name] = values
        if where is not None:
            out.point_data['vtkValidPointMask'] = where.astype(np.uint8)
        return out

    def _check_memory(self, n_points: int, max_memory: float) -> None:
        """Raise before PyKrige allocates a distance matrix larger than ``max_memory``."""
        needed = n_points * (self._n_samples + 1) * 8
        if needed > max_memory:
            msg = (
                f'Kriging {n_points} points against {self._n_samples} samples needs about '
                f'{needed / 1e9:.2f} GB for the distance matrix, more than max_memory '
                f'({max_memory / 1e9:.2f} GB). Pass n_closest_points to krige with a moving '
                'window, use a coarser target, or raise max_memory (None disables this check).'
            )
            raise ValueError(msg)

    def _execute_kwargs(self, backend: str | None, n_closest_points: int | None) -> dict:
        """Validate backend options up front and return PyKrige ``execute`` kwargs."""
        if backend not in (None, 'vectorized', 'loop', 'C'):
            msg = f"backend must be one of 'vectorized', 'loop', 'C', got {backend!r}."
            raise ValueError(msg)
        ordinary_2d = self._method == 'ordinary' and self._dim == 2
        if backend == 'C' and not ordinary_2d:
            msg = "backend 'C' is only available for 2D ordinary kriging."
            raise ValueError(msg)
        if n_closest_points is None:
            return {'backend': backend or 'vectorized'}
        if self._method == 'universal':
            msg = 'n_closest_points is not supported for universal kriging.'
            raise ValueError(msg)
        if n_closest_points < 2:
            msg = f'n_closest_points must be at least 2, got {n_closest_points}.'
            raise ValueError(msg)
        backend = backend or 'loop'
        if backend == 'vectorized':
            msg = "n_closest_points requires backend 'loop' (or 'C' for 2D)."
            raise ValueError(msg)
        return {'backend': backend, 'n_closest_points': n_closest_points}


def _kriging_class(method: str, dim: int) -> type:
    """Return the PyKrige class for ``method`` in ``dim`` dimensions."""
    from pykrige.ok import OrdinaryKriging  # noqa: PLC0415
    from pykrige.ok3d import OrdinaryKriging3D  # noqa: PLC0415
    from pykrige.uk import UniversalKriging  # noqa: PLC0415
    from pykrige.uk3d import UniversalKriging3D  # noqa: PLC0415

    classes = {
        ('ordinary', 2): OrdinaryKriging,
        ('ordinary', 3): OrdinaryKriging3D,
        ('universal', 2): UniversalKriging,
        ('universal', 3): UniversalKriging3D,
    }
    return classes[method, dim]
