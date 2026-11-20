"""Fitted kriging model that predicts onto PyVista datasets."""

from __future__ import annotations

import copy
from typing import Any
from typing import NamedTuple

import numpy as np
import pyvista as pv

from pyvista_krige._advanced import detrended
from pyvista_krige._advanced import drift_matrix
from pyvista_krige._advanced import imply_drift_terms
from pyvista_krige._advanced import resolve_dim
from pyvista_krige._advanced import translate_drift
from pyvista_krige._anisotropy import estimate_anisotropy
from pyvista_krige._simulate import simulate
from pyvista_krige._suggest import pick_samples
from pyvista_krige._targets import execute_on
from pyvista_krige._targets import near_samples
from pyvista_krige._targets import resolve_where
from pyvista_krige._targets import sample_arrays
from pyvista_krige._transform import Transform
from pyvista_krige._transform import fit_transform
from pyvista_krige._variogram import ExperimentalVariogram
from pyvista_krige._variogram import Variogram
from pyvista_krige._variogram import experimental_variogram
from pyvista_krige._variogram import fit_model
from pyvista_krige._variogram import parameter_dict

# PyKrige's anisotropy options of 2D and 3D models, as attributes of its instances.
_ANISOTROPY_2D = ('anisotropy_scaling', 'anisotropy_angle')
_ANISOTROPY_3D = (
    'anisotropy_scaling_y',
    'anisotropy_scaling_z',
    'anisotropy_angle_x',
    'anisotropy_angle_y',
    'anisotropy_angle_z',
)

# Options of PyKrige's constructors that ``update_variogram_model`` also takes.
_VARIOGRAM_OPTIONS = ('variogram_parameters', 'variogram_function', 'nlags', 'weight')


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


class HoldOut(NamedTuple):
    """Cross-validation errors of a kriging model, from :meth:`KrigingModel.cross_validate`.

    Attributes
    ----------
    rmse : float
        Root-mean-square hold-out error.

    mae : float
        Mean absolute hold-out error.

    bias : float
        Mean hold-out error (estimate minus observed); ideally close to 0.

    samples : pyvista.PolyData
        The samples, with ``'<scalars> error'`` (estimate minus observed) and
        ``'<scalars> z-score'`` (error over the kriging standard deviation; its
        spread is ideally close to 1) point arrays.

    scalars : str
        Name of the sampled array.

    """

    rmse: float
    mae: float
    bias: float
    samples: pv.PolyData
    scalars: str


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
        samples: pv.PolyData | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        experimental: ExperimentalVariogram | None = None,
        transform: Transform | None = None,
    ) -> None:
        self._instance = instance
        self._scalars = scalars
        self._method = method
        self._dim = dim
        self._n_samples = n_samples
        self._drift_names = drift_names
        self._samples = samples
        self._fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        self._experimental = experimental
        self._transform = transform

    @classmethod
    def fit(
        cls,
        samples: pv.DataSet,
        scalars: str,
        *,
        method: str = 'ordinary',
        dim: int | None = None,
        transform: str | None = None,
        transform_limits: tuple[float | None, float | None] | None = None,
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

        transform : str, optional
            ``'log'`` or ``'normal_score'`` to krige transformed values.

        transform_limits : tuple[float | None, float | None], optional
            Lower and upper limits of the normal score distribution.

        **kwargs : dict, optional
            Options forwarded to the PyKrige class, plus ``detrend``,
            ``max_lag`` and ``anisotropy``.

        Returns
        -------
        KrigingModel
            The fitted model.

        """
        fit_kwargs = dict(kwargs)
        if transform is not None:
            fit_kwargs.update(transform=transform, transform_limits=transform_limits)
        detrend = kwargs.pop('detrend', False)
        max_lag = kwargs.pop('max_lag', None)
        anisotropy = kwargs.pop('anisotropy', None)
        if anisotropy is not None and not (isinstance(anisotropy, str) and anisotropy == 'auto'):
            msg = f"anisotropy must be None or 'auto', got {anisotropy!r}."
            raise ValueError(msg)
        found = sample_arrays(samples, scalars)
        points, values = found.points, found.values
        fitted_transform = fit_transform(transform, values, transform_limits)
        if fitted_transform is not None:
            values = fitted_transform.forward(values)
        drift_names = translate_drift(found, kwargs)
        if method not in ('ordinary', 'universal'):
            msg = f"method must be 'ordinary' or 'universal', got {method!r}."
            raise ValueError(msg)
        if detrend and method != 'universal':
            msg = "detrend needs universal kriging: pass method='universal' and a drift."
            raise ValueError(msg)
        if method == 'universal':
            imply_drift_terms(kwargs)
        dim = resolve_dim(points, kwargs, dim, method)
        if anisotropy is not None:
            kwargs.update(_estimate_anisotropy(points[:, :dim], values, kwargs, detrend, max_lag))
        kriging_class = _kriging_class(method, dim)
        instance = kriging_class(*points.T[:dim], values, **kwargs)
        experimental = None
        if detrend or max_lag is not None:
            instance, experimental = _refit_variogram(
                kriging_class, instance, points[:, :dim], values, kwargs, detrend, max_lag
            )
        return cls(
            instance,
            scalars,
            method=method,
            dim=dim,
            n_samples=len(values),
            drift_names=drift_names,
            samples=_sample_cloud(found, scalars, drift_names),
            fit_kwargs=fit_kwargs,
            experimental=experimental,
            transform=fitted_transform,
        )

    @property
    def dim(self) -> int:
        """Return the spatial dimension of the model, 2 or 3."""
        return self._dim

    @property
    def n_samples(self) -> int:
        """Return the number of samples the model was fitted to."""
        return self._n_samples

    @property
    def samples(self) -> pv.PolyData | None:
        """Return the fitted samples as points with the sampled and drift arrays.

        The sampled values are in their original units, before any transform.
        """
        return self._samples

    @property
    def transform(self) -> Transform | None:
        """Return the transform the values were kriged in, or ``None``.

        A :class:`LogTransform` or :class:`NormalScoreTransform`, fitted to the
        samples. Its ``forward`` maps values to the kriged space and its
        ``back`` maps an estimate and variance back to the original units.
        """
        return self._transform

    @property
    def variogram(self) -> Variogram:
        """Return the experimental variogram and the fitted model.

        With ``detrend=True`` the experimental variogram is that of the drift
        residuals, which is what the model was fitted to. With a ``transform``
        it is that of the transformed values.
        """
        instance = self._instance
        lags, semivariance = instance.lags, instance.semivariance
        counts = None
        if self._experimental is not None:
            lags, semivariance = self._experimental.lags, self._experimental.semivariance
            counts = self._experimental.counts
        return Variogram(
            model_name=instance.variogram_model,
            parameters=np.asarray(instance.variogram_model_parameters, dtype=float),
            lags=np.asarray(lags, dtype=float),
            semivariance=np.asarray(semivariance, dtype=float),
            function=instance.variogram_function,
            counts=counts,
        )

    @property
    def anisotropy(self) -> dict[str, float]:
        """Return the anisotropy options PyKrige applies, whether given or estimated.

        A 2D model has ``anisotropy_scaling`` and ``anisotropy_angle``. A 3D
        model has ``anisotropy_scaling_y``, ``anisotropy_scaling_z`` and
        ``anisotropy_angle_x``/``_y``/``_z``. Pass them to
        :meth:`KrigeAccessor.fit` to reuse them without ``anisotropy='auto'``.
        """
        names = _ANISOTROPY_3D if self._dim == 3 else _ANISOTROPY_2D
        return {name: float(getattr(self._instance, name)) for name in names}

    @property
    def statistics(self) -> Statistics:
        """Return PyKrige's cross-validation statistics of the fit.

        With a ``transform`` they are computed on the transformed values.

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
            ``variogram_parameters`` or ``nlags``. Anisotropy, ``nlags`` and
            ``weight`` default to this model's. An ``anisotropy='auto'``
            estimate is kept as it is, not estimated again.

        Returns
        -------
        KrigingModel
            The refitted model. This model is unchanged.

        """
        options = {k: v for k, v in self._fit_kwargs.items() if k in _VARIOGRAM_OPTIONS}
        options.pop('variogram_parameters', None)
        options.pop('variogram_function', None)
        # Read from the instance so an estimated anisotropy carries over.
        options.update(self.anisotropy)
        options.update(kwargs)
        instance = copy.deepcopy(self._instance)
        instance.update_variogram_model(variogram_model, **options)
        fit_kwargs = {**self._fit_kwargs}
        if fit_kwargs.pop('anisotropy', None) is not None:
            # The estimate belongs to the model it was made for, so cross_validate keeps it.
            fit_kwargs.update(self.anisotropy)
        fit_kwargs.update(variogram_model=variogram_model, **kwargs)
        for key in ('detrend', 'max_lag'):
            fit_kwargs.pop(key, None)
        return KrigingModel(
            instance,
            self._scalars,
            method=self._method,
            dim=self._dim,
            n_samples=self._n_samples,
            drift_names=self._drift_names,
            samples=self._samples,
            fit_kwargs=fit_kwargs,
            transform=self._transform,
        )

    def cross_validate(
        self,
        folds: int | str = 5,
        *,
        seed: int = 0,
        **kwargs: Any,
    ) -> HoldOut:
        """Score this model by refitting it with each fold of samples held out.

        Every fold refits the model with the same options, including
        ``detrend`` and ``max_lag``, and predicts the held-out samples. Works for
        every method and dimension, and needs no extra packages. With a
        ``transform``, each fold refits the transform to its own samples, and
        the errors are in the original units.

        Parameters
        ----------
        folds : int | str, default: 5
            Number of folds, assigned at random, or ``'loo'`` to hold out one
            sample at a time.

        seed : int, default: 0
            Seed for the random fold assignment.

        **kwargs : dict, optional
            Options forwarded to :meth:`predict`, such as ``n_closest_points``.

        Returns
        -------
        HoldOut
            Error statistics and the samples with per-sample errors.

        """
        samples, scalars, n = self._samples, self._scalars, self._n_samples
        if folds == 'loo':
            assignment = np.arange(n)
        elif isinstance(folds, (int, np.integer)) and 2 <= folds <= n:
            assignment = np.random.default_rng(seed).permutation(n) % folds
        else:
            msg = f"folds must be 'loo' or an integer from 2 to {n}, got {folds!r}."
            raise ValueError(msg)
        estimate = np.empty(n)
        variance = np.empty(n)
        for fold in np.unique(assignment):
            test = assignment == fold
            model = KrigingModel.fit(
                _subset(samples, ~test),
                scalars,
                method=self._method,
                dim=self._dim,
                **self._fit_kwargs,
            )
            out = model.predict(_subset(samples, test), **kwargs)
            estimate[test] = out[scalars]
            variance[test] = out[f'{scalars} variance']
        out = samples.copy()
        error = estimate - samples[scalars]
        std = np.sqrt(np.clip(variance, 0, None))
        out[f'{scalars} error'] = error
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f'{scalars} z-score'] = error / std
        return HoldOut(
            rmse=float(np.sqrt(np.mean(error**2))),
            mae=float(np.mean(np.abs(error))),
            bias=float(np.mean(error)),
            samples=out,
            scalars=scalars,
        )

    def suggest_samples(
        self,
        target: pv.DataSet,
        n: int,
        *,
        where: str | np.ndarray | None = None,
        min_spacing: float | None = None,
    ) -> pv.PolyData:
        """Suggest where to take the next ``n`` samples, where the kriging error is largest.

        Picks greedily. Each step krigs the variance onto the candidate points,
        takes the point with the largest one, and adds it as a pseudo-sample before
        the next step. Kriging variance does not depend on the sampled values, so
        the pseudo-sample takes the current estimate. The variogram is held fixed,
        not refitted, and every step krigs from all samples with a full kriging
        system, so ``n_closest_points`` windows are not used.

        Parameters
        ----------
        target : pyvista.DataSet
            Dataset whose points are the candidates. It is not modified.

        n : int
            Number of points to pick.

        where : str | numpy.ndarray, optional
            Boolean array, or name of a target point array, that is ``True``
            at the candidate points. Defaults to every point.

        min_spacing : float, optional
            Drop candidates closer than this to a pick, in the model's coordinates
            (x and y only for a 2D model). Without it, picks can cluster
            where the variance is high.

        Returns
        -------
        pyvista.PolyData
            The picked points, in pick order, with these point arrays:
            ``'pick order'`` (1 for the first pick) and ``'std before pick'``
            (the kriging standard deviation at the point before it was picked).

        Raises
        ------
        TypeError
            If ``target`` is a :class:`pyvista.MultiBlock`.

        ValueError
            If ``n`` is less than 1 or more than the number of candidates, or
            ``min_spacing`` is negative.

        Warns
        -----
        UserWarning
            If ``min_spacing`` leaves no candidates before ``n`` picks are made.
            The picks made so far are returned.

        Notes
        -----
        Universal kriging reads the drift values of the picks from the ``specified_drift``
        arrays of ``target``, as :meth:`predict` does.

        """
        return pick_samples(self, target, n, where=where, min_spacing=min_spacing)

    def simulate(
        self,
        target: pv.DataSet,
        n: int,
        *,
        seed: int | np.random.Generator | None = None,
        n_neighbors: int = 16,
        summary: bool = False,
        exceed: float | None = None,
    ) -> pv.DataSet:
        """Draw ``n`` conditional realizations onto the points of ``target``.

        Kriging gives one smooth surface. A realization is one equally likely
        field that varies like the data and equals the samples at their
        locations. Many realizations, summarized point by point, give a mean,
        percentiles and exceedance probabilities that do not assume normal
        errors in the original units.

        Uses sequential Gaussian simulation in normal score space. The target
        points are visited in a random path. Each is kriged from its
        ``n_neighbors`` nearest samples and earlier points by ordinary kriging,
        drawn from the normal distribution of that estimate and variance, and
        added as conditioning data. Each realization is then back-transformed to
        the original units.

        Parameters
        ----------
        target : pyvista.DataSet
            Dataset whose points are simulated. It is not modified.

        n : int
            Number of realizations.

        seed : int | numpy.random.Generator, optional
            Seed for the random path and draws. The same seed gives the same
            realizations.

        n_neighbors : int, default: 16
            Number of nearest samples and earlier points each point is kriged
            from, in PyKrige's anisotropy-adjusted coordinates.

        summary : bool, default: False
            Write ``'<scalars> mean'``, ``'<scalars> P10'``, ``'<scalars> P50'``
            and ``'<scalars> P90'`` instead of the realizations.

        exceed : float, optional
            Write the fraction of realizations above this value as
            ``'P(<scalars> > <exceed>)'`` instead of the realizations.

        Returns
        -------
        pyvista.DataSet
            Copy of ``target`` with the realizations as point arrays
            ``'<scalars> realization 0'`` to ``'<scalars> realization <n - 1>'``,
            or the summaries asked for.

        Raises
        ------
        TypeError
            If ``target`` is a :class:`pyvista.MultiBlock`.

        ValueError
            If the model is universal or uses geographic coordinates, or ``n`` or
            ``n_neighbors`` is less than 1.

        Notes
        -----
        A model fitted with ``transform='normal_score'`` is simulated with its
        transform and variogram. Any other model gets a normal score transform
        fitted to its samples, and its variogram divided by the variance of the
        kriged sample values so that it applies to unit-variance scores.

        All realizations share one random path, so each point's kriging weights
        are solved once and extra realizations cost little. Target points at a
        sample take the sample value, with or without a nugget.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> samples = pv.PolyData(rng.uniform(0, 10, (30, 3)) * [1, 1, 0])
        >>> samples['grade'] = np.exp(
        ...     samples.points[:, 0] / 5 + rng.normal(0, 0.3, 30)
        ... )
        >>> model = samples.krige.fit('grade', variogram_model='exponential')
        >>> grid = pv.ImageData(dimensions=(20, 20, 1), spacing=(0.5, 0.5, 1))
        >>> result = model.simulate(grid, n=10, seed=0)
        >>> result['grade realization 9'].shape
        (400,)
        >>> summary = model.simulate(grid, n=100, seed=0, summary=True)
        >>> bool(np.all(summary['grade P10'] <= summary['grade P90']))
        True

        """
        return simulate(
            self,
            target,
            n,
            seed=seed,
            n_neighbors=n_neighbors,
            summary=summary,
            exceed=exceed,
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
        max_distance: float | None = None,
        max_std: float | None = None,
        null_value: float = np.nan,
    ) -> pv.DataSet:
        """Krige onto the points of ``target`` and return a copy with the estimate.

        Axis-aligned :class:`pyvista.ImageData` and :class:`pyvista.RectilinearGrid`
        targets use PyKrige's fast grid path; every other dataset is kriged
        point by point.

        With a ``transform``, the estimate and variance are the mean and
        variance of the back-transformed distribution, in the original units.
        The kriged values in the transformed space are also written, as
        ``'<scalars> transformed'`` and ``'<scalars> transformed variance'``
        (or ``std``), for :func:`pyvista_krige.exceedance`.

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
            Without ``n_closest_points`` PyKrige builds a dense matrix between
            targets and samples. Krige in chunks small enough to keep that
            matrix under this many bytes. Grids are split into slabs along z
            (y in 2D) so they keep the fast grid path. ``None`` kriges every
            point in one call.

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

        max_distance : float, optional
            Skip points farther than this from every sample, in the model's
            coordinates (x and y only for a 2D model, without anisotropy).
            Combines with ``where``. Use it to avoid extrapolating.

        max_std : float, optional
            Null points whose kriging standard deviation exceeds this, after
            kriging and in the original units. Combines with ``where`` and
            ``max_distance``.

        null_value : float, default: nan
            Value written to points excluded by ``where``, ``max_distance`` or
            ``max_std``.

        Returns
        -------
        pyvista.DataSet
            Copy of ``target`` with the estimate in ``point_data[scalars]``.

        Raises
        ------
        TypeError
            If ``target`` is a :class:`pyvista.MultiBlock`.

        ValueError
            If the backend options are unsupported for this model.

        """
        if isinstance(target, pv.MultiBlock):
            msg = 'MultiBlock targets are not supported. Use target.combine() first.'
            raise TypeError(msg)
        execute_kwargs = self._execute_kwargs(backend, n_closest_points)
        where = resolve_where(target, where)
        if max_distance is not None:
            near = near_samples(self._samples.points, target, self._dim, max_distance)
            where = near if where is None else where & near
        max_points = None
        if n_closest_points is None and max_memory is not None:
            # One float64 per target point and kriging-matrix column.
            max_points = max(1, int(max_memory // ((self._n_samples + 1) * 8)))
        drift_arrays = None
        if self._drift_names is not None:
            drift_arrays = [
                np.asarray(target.point_data[name], dtype=float) for name in self._drift_names
            ]
        estimate, sigma2 = execute_on(
            self._instance, target, self._dim, where, drift_arrays, max_points, **execute_kwargs
        )
        transformed = None
        if self._transform is not None:
            transformed = estimate, sigma2
            estimate, sigma2 = self._transform.back(estimate, sigma2)
        if max_std is not None:
            # NaN compares False, so points left unkriged count as uncertain.
            certain = np.sqrt(np.clip(sigma2, 0, None)) <= max_std
            where = certain if where is None else where & certain
        out = target.copy()
        arrays = {self._scalars: estimate}
        if variance:
            name = variance if isinstance(variance, str) else _uncertainty_name(self._scalars, std)
            arrays[name] = _uncertainty(sigma2, std)
        if transformed is not None:
            name = f'{self._scalars} transformed'
            arrays[name] = transformed[0]
            if variance:
                arrays[_uncertainty_name(name, std)] = _uncertainty(transformed[1], std)
        for name, values in arrays.items():
            if where is not None:
                values[~where] = null_value
            out.point_data[name] = values
        if where is not None:
            out.point_data['vtkValidPointMask'] = where.astype(np.uint8)
        return out

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


def _uncertainty_name(scalars: str, std: bool) -> str:  # noqa: FBT001
    """Return the default name of the variance or standard deviation of ``scalars``."""
    return f'{scalars} {"std" if std else "variance"}'


def _uncertainty(sigma2: np.ndarray, std: bool) -> np.ndarray:  # noqa: FBT001
    """Return the kriging variance, or its square root when ``std`` is set."""
    # Round-off can leave tiny negative variances at the samples.
    return np.sqrt(np.clip(sigma2, 0, None)) if std else sigma2


def _sample_cloud(found: Any, scalars: str, drift_names: list[str] | None) -> pv.PolyData:
    """Return the finite samples as points carrying the sampled and drift arrays."""
    cloud = pv.PolyData(found.points.copy())
    cloud.point_data[scalars] = found.values
    for name in drift_names or []:
        cloud.point_data[name] = found.aligned(name)
    return cloud


def _subset(cloud: pv.PolyData, mask: np.ndarray) -> pv.PolyData:
    """Return the points of ``cloud`` selected by ``mask``, with their point arrays."""
    part = pv.PolyData(cloud.points[mask])
    for name in cloud.point_data:
        part.point_data[name] = cloud.point_data[name][mask]
    return part


def _estimate_anisotropy(
    coords: np.ndarray,
    values: np.ndarray,
    kwargs: dict[str, Any],
    detrend: bool,  # noqa: FBT001
    max_lag: float | None,
) -> dict[str, float]:
    """Return PyKrige anisotropy options estimated for ``anisotropy='auto'``."""
    variogram_model = kwargs.get('variogram_model', 'linear')
    if not isinstance(variogram_model, str) or variogram_model == 'custom':
        msg = (
            "anisotropy='auto' needs a named PyKrige variogram model, not a GSTools CovModel "
            "or 'custom' function. Set a CovModel's anisotropy on the CovModel itself."
        )
        raise ValueError(msg)
    if kwargs.get('coordinates_type', 'euclidean') == 'geographic':
        msg = "anisotropy='auto' does not support coordinates_type='geographic'."
        raise ValueError(msg)
    given = sorted(set(kwargs) & {*_ANISOTROPY_2D, *_ANISOTROPY_3D})
    if given:
        msg = f"anisotropy='auto' estimates {given}, so do not pass them as well."
        raise ValueError(msg)
    if detrend:
        # Center the coordinates as PyKrige does before evaluating the drift.
        center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        values = detrended(values, drift_matrix(coords - center, kwargs))
    return estimate_anisotropy(
        coords, values, variogram_model, nlags=kwargs.get('nlags', 6), max_lag=max_lag
    )


def _refit_variogram(
    kriging_class: type,
    instance: Any,
    coords: np.ndarray,
    values: np.ndarray,
    kwargs: dict[str, Any],
    detrend: bool,  # noqa: FBT001
    max_lag: float | None,
) -> tuple[Any, ExperimentalVariogram]:
    """Refit the variogram to the drift residuals and/or lags up to ``max_lag``.

    The experimental variogram is computed in PyKrige's anisotropy-adjusted
    coordinates, where its variogram model applies. Returns the refitted PyKrige
    object and the experimental variogram it was fitted to.
    """
    variogram_model = kwargs.get('variogram_model', 'linear')
    if not isinstance(variogram_model, str) or variogram_model == 'custom':
        msg = (
            'detrend and max_lag fit the variogram themselves, so they need a named PyKrige '
            'model, not a GSTools CovModel or a custom function.'
        )
        raise ValueError(msg)
    adjusted = [instance.X_ADJUSTED, instance.Y_ADJUSTED]
    if coords.shape[1] == 3:
        adjusted.append(instance.Z_ADJUSTED)
    adjusted = np.column_stack(adjusted)
    residuals = detrended(values, drift_matrix(adjusted, kwargs)) if detrend else values
    if max_lag is None:
        # PyKrige's own fit spans every pair distance.
        max_lag = float(np.linalg.norm(np.ptp(adjusted, axis=0)))
    experimental = experimental_variogram(
        adjusted, residuals, nlags=kwargs.get('nlags', 6), max_lag=max_lag
    )
    if kwargs.get('variogram_parameters') is not None:
        return instance, experimental
    parameters = fit_model(
        variogram_model,
        experimental.lags,
        experimental.semivariance,
        experimental.counts,
        experimental.max_lag,
    )
    refit_kwargs = {**kwargs, 'variogram_parameters': parameter_dict(variogram_model, parameters)}
    return kriging_class(*coords.T, values, **refit_kwargs), experimental


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
