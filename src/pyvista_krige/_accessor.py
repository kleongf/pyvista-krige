"""Register the ``krige`` accessor on :class:`pyvista.DataSet`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv

from pyvista_krige._targets import image_spanning

if TYPE_CHECKING:
    from pyvista_krige._advanced import CrossValidation
    from pyvista_krige._model import KrigingModel
    from pyvista_krige._variogram import ExperimentalVariogram


@pv.register_dataset_accessor('krige', pv.DataSet)
class KrigeAccessor:
    """Kriging methods for scattered samples stored in a dataset.

    Parameters
    ----------
    dataset : pyvista.DataSet
        Dataset whose points carry the sampled values.

    """

    def __init__(self, dataset: pv.DataSet) -> None:
        self._dataset = dataset

    def fit(
        self,
        scalars: str,
        *,
        method: str = 'ordinary',
        dim: int | None = None,
        **kwargs: Any,
    ) -> KrigingModel:
        """Fit a kriging model to the ``scalars`` array of this dataset.

        Point arrays are sampled at the points and cell arrays at the cell
        centers. Samples with NaN or infinite values are dropped with a warning.

        Parameters
        ----------
        scalars : str
            Name of the point or cell data array holding the sampled values.

        method : str, default: 'ordinary'
            ``'ordinary'`` or ``'universal'`` kriging.

        dim : int, optional
            Spatial dimension, 2 or 3. By default a GSTools ``variogram_model``
            decides, otherwise samples with a spread in z are kriged in 3D and
            flat samples in 2D. A 2D model uses only the x and y coordinates.

        **kwargs : dict, optional
            Options forwarded to the PyKrige class (``OrdinaryKriging``,
            ``UniversalKriging``, ``OrdinaryKriging3D`` or ``UniversalKriging3D``):

            - ``variogram_model``: ``'linear'``, ``'power'``, ``'gaussian'``,
              ``'spherical'``, ``'exponential'``, ``'hole-effect'``,
              ``'custom'`` (with ``variogram_function``), or a
              ``gstools.CovModel``.
            - ``variogram_parameters``, ``nlags``, ``weight``: control the
              variogram fit.
            - ``anisotropy_scaling`` and ``anisotropy_angle`` (2D), or
              ``anisotropy_scaling_y``/``_z`` and ``anisotropy_angle_x``/``_y``/``_z``
              (3D). PyKrige and GSTools define the angles differently.
            - ``exact_values``, ``pseudo_inv``, ``pseudo_inv_type``.
            - ``coordinates_type='geographic'`` for longitude/latitude samples
              (2D ordinary kriging only).
            - ``enable_statistics=True`` to compute :attr:`KrigingModel.statistics`
              for 2D ordinary kriging.
            - Universal kriging drift: ``drift_terms``, ``point_drift``,
              ``functional_drift``, and two dataset-aware options:
              ``specified_drift`` names arrays present on these samples and on
              every prediction target; ``external_drift`` may be a single-layer
              :class:`pyvista.ImageData` with the drift in the array named by
              ``external_drift_scalars``. Giving a drift option adds its term
              to ``drift_terms``, which PyKrige otherwise requires.

            Two options are handled here rather than by PyKrige:

            - ``detrend=True`` (universal kriging): fit the variogram to the
              least-squares residuals of the drift instead of the raw values.
              PyKrige fits the raw values, so a strong trend inflates the sill
              and range. Supports the ``regional_linear``, ``specified`` and
              ``functional`` drift terms.
            - ``max_lag``: fit the variogram only to pairs closer than this,
              since long lags have few pairs and often show the trend rather
              than the correlation. Half the sample extent is a common choice.

        Returns
        -------
        KrigingModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> samples = pv.PolyData(rng.uniform(0, 10, (30, 3)) * [1, 1, 0])
        >>> samples['height'] = np.sin(samples.points[:, 0])
        >>> model = samples.krige.fit('height', variogram_model='spherical')
        >>> model.dim
        2

        """
        from pyvista_krige._model import KrigingModel  # noqa: PLC0415

        return KrigingModel.fit(self._dataset, scalars, method=method, dim=dim, **kwargs)

    def to_image(
        self,
        scalars: str,
        *,
        dimensions: tuple[int, ...] | None = None,
        spacing: float | tuple[float, ...] | None = None,
        padding: float = 0.0,
        backend: str | None = None,
        n_closest_points: int | None = None,
        max_memory: float | None = 1e9,
        variance: str | bool = True,
        std: bool = False,
        max_distance: float | None = None,
        max_std: float | None = None,
        **kwargs: Any,
    ) -> pv.ImageData:
        """Krige ``scalars`` onto a new image spanning the bounds of this dataset.

        Parameters
        ----------
        scalars : str
            Name of the point or cell data array holding the sampled values.

        dimensions : sequence[int], optional
            Number of points along each axis. Defaults to 50 along each
            kriged axis. Mutually exclusive with ``spacing``.

        spacing : float | sequence[float], optional
            Distance between image points. Mutually exclusive with
            ``dimensions``.

        padding : float, default: 0.0
            Grow the bounds by this fraction of the extent on each side.

        backend : str, optional
            PyKrige backend. See :meth:`KrigingModel.predict`.

        n_closest_points : int, optional
            Krige each point from its nearest samples only.

        max_memory : float | None, default: 1e9
            Krige in chunks whose kriging matrix stays under this many bytes.

        variance : str | bool, default: True
            Write the kriging variance, optionally under this name.

        std : bool, default: False
            Write the standard deviation instead of the variance.

        max_distance : float, optional
            Skip image points farther than this from every sample.

        max_std : float, optional
            Null image points whose kriging standard deviation exceeds this.

        **kwargs : dict, optional
            Options forwarded to :meth:`fit`.

        Returns
        -------
        pyvista.ImageData
            Image with the estimate and its variance as point data.

        """
        model = self.fit(scalars, **kwargs)
        image = image_spanning(
            self._dataset.bounds,
            model.dim,
            dimensions=dimensions,
            spacing=spacing,
            padding=padding,
        )
        return model.predict(
            image,
            backend=backend,
            n_closest_points=n_closest_points,
            max_memory=max_memory,
            variance=variance,
            std=std,
            max_distance=max_distance,
            max_std=max_std,
        )

    def variogram(
        self,
        scalars: str,
        *,
        nlags: int = 12,
        max_lag: float | None = None,
        direction: str | tuple[float, ...] | None = None,
        tolerance: float = 22.5,
        bandwidth: float | None = None,
        dim: int | None = None,
        detrend: bool = False,
        **kwargs: Any,
    ) -> ExperimentalVariogram:
        """Compute the experimental variogram of ``scalars``, optionally along a direction.

        Comparing directions shows anisotropy. For example, a vertical range
        a third of the horizontal one suggests ``anisotropy_scaling_z=3`` when
        fitting. Plot the result with ``pl.krige.add_variogram``.

        Parameters
        ----------
        scalars : str
            Name of the point or cell data array holding the sampled values.

        nlags : int, default: 12
            Number of equal-width lag bins between 0 and ``max_lag``.

        max_lag : float, optional
            Largest pair distance used. Defaults to half the diagonal of the
            samples' bounding box.

        direction : str | sequence[float], optional
            ``'x'``, ``'y'``, ``'z'`` (or ``'vertical'``), ``'horizontal'``, or
            a direction vector. By default pairs in every direction are used.

        tolerance : float, default: 22.5
            Angular tolerance in degrees around ``direction``. For
            ``'horizontal'``, the largest angle out of the xy plane.

        bandwidth : float, optional
            Largest distance of a pair from the ``direction`` line.

        dim : int, optional
            Spatial dimension, 2 or 3. Detected from the samples by default.

        detrend : bool, default: False
            Use the least-squares residuals of the drift given in ``kwargs``.

        **kwargs : dict, optional
            Drift options as for :meth:`fit`: ``drift_terms``,
            ``specified_drift`` and ``functional_drift``.

        Returns
        -------
        ExperimentalVariogram
            Lags, semivariance and pair counts of the non-empty bins.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> samples = pv.PolyData(rng.uniform(0, 10, (200, 3)))
        >>> samples['layered'] = np.sin(samples.points[:, 2])
        >>> vertical = samples.krige.variogram('layered', direction='vertical')
        >>> horizontal = samples.krige.variogram('layered', direction='horizontal')
        >>> bool(vertical.semivariance.max() > horizontal.semivariance.max())
        True

        """
        from pyvista_krige._advanced import detrended  # noqa: PLC0415
        from pyvista_krige._advanced import drift_matrix  # noqa: PLC0415
        from pyvista_krige._advanced import imply_drift_terms  # noqa: PLC0415
        from pyvista_krige._advanced import resolve_dim  # noqa: PLC0415
        from pyvista_krige._advanced import translate_drift  # noqa: PLC0415
        from pyvista_krige._targets import sample_arrays  # noqa: PLC0415
        from pyvista_krige._variogram import experimental_variogram  # noqa: PLC0415

        found = sample_arrays(self._dataset, scalars)
        dim = resolve_dim(found.points, {}, dim, 'ordinary')
        coords = found.points[:, :dim]
        values = found.values
        if detrend:
            translate_drift(found, kwargs)
            imply_drift_terms(kwargs)
            # Center the coordinates as PyKrige does before evaluating the drift.
            center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
            values = detrended(values, drift_matrix(coords - center, kwargs))
        elif kwargs:
            msg = f'Drift options {sorted(kwargs)} only apply with detrend=True.'
            raise TypeError(msg)
        return experimental_variogram(
            coords,
            values,
            nlags=nlags,
            max_lag=max_lag,
            direction=direction,
            tolerance=tolerance,
            bandwidth=bandwidth,
        )

    def regression(
        self,
        target: pv.DataSet,
        scalars: str,
        *,
        predictors: list[str],
        regression_model: object | None = None,
        method: str = 'ordinary',
        dim: int | None = None,
        **kwargs: Any,
    ) -> pv.DataSet:
        """Predict ``scalars`` on ``target`` with regression kriging.

        A scikit-learn regressor is fitted to the ``predictors`` arrays, and its
        residuals are kriged. Requires the ``sklearn`` extra.

        Parameters
        ----------
        target : pyvista.DataSet
            Dataset to predict onto. It must carry the ``predictors`` arrays.

        scalars : str
            Name of the point data array holding the sampled values.

        predictors : list[str]
            Names of point data arrays present on both datasets.

        regression_model : sklearn.base.RegressorMixin, optional
            Regressor to fit. Defaults to PyKrige's ``SVR()``.

        method : str, default: 'ordinary'
            Kriging method for the residuals, ``'ordinary'`` or ``'universal'``.

        dim : int, optional
            Spatial dimension, 2 or 3. Detected from the samples by default.

        **kwargs : dict, optional
            Options forwarded to :class:`pykrige.rk.RegressionKriging`, such as
            ``variogram_model`` or ``n_closest_points``.

        Returns
        -------
        pyvista.DataSet
            Copy of ``target`` with the prediction in ``point_data[scalars]``.

        """
        from pyvista_krige._advanced import regression_kriging  # noqa: PLC0415

        return regression_kriging(
            self._dataset,
            target,
            scalars,
            predictors=predictors,
            model=regression_model,
            method=method,
            dim=dim,
            classify=False,
            **kwargs,
        )

    def classify(
        self,
        target: pv.DataSet,
        scalars: str,
        *,
        predictors: list[str],
        classification_model: object | None = None,
        method: str = 'ordinary',
        dim: int | None = None,
        **kwargs: Any,
    ) -> pv.DataSet:
        """Predict class labels ``scalars`` on ``target`` with classification kriging.

        A scikit-learn classifier is fitted to the ``predictors`` arrays, and its
        class probabilities are corrected by kriging. Requires the ``sklearn``
        extra.

        Parameters
        ----------
        target : pyvista.DataSet
            Dataset to predict onto. It must carry the ``predictors`` arrays.

        scalars : str
            Name of the integer point data array holding the sampled labels.

        predictors : list[str]
            Names of point data arrays present on both datasets.

        classification_model : sklearn.base.ClassifierMixin, optional
            Classifier to fit. Defaults to PyKrige's ``SVC()``.

        method : str, default: 'ordinary'
            Kriging method, ``'ordinary'`` or ``'universal'``.

        dim : int, optional
            Spatial dimension, 2 or 3. Detected from the samples by default.

        **kwargs : dict, optional
            Options forwarded to :class:`pykrige.ck.ClassificationKriging`.

        Returns
        -------
        pyvista.DataSet
            Copy of ``target`` with the labels in ``point_data[scalars]``.

        """
        from pyvista_krige._advanced import regression_kriging  # noqa: PLC0415

        return regression_kriging(
            self._dataset,
            target,
            scalars,
            predictors=predictors,
            model=classification_model,
            method=method,
            dim=dim,
            classify=True,
            **kwargs,
        )

    def cross_validate(
        self,
        scalars: str,
        param_grid: dict[str, list[Any]],
        *,
        cv: int = 5,
        scoring: str | None = None,
        method: str = 'ordinary',
        dim: int | None = None,
        **kwargs: Any,
    ) -> CrossValidation:
        """Choose kriging parameters by cross-validation and refit the best model.

        Runs scikit-learn's ``GridSearchCV`` over :class:`pykrige.rk.Krige`.
        Requires the ``sklearn`` extra.

        Parameters
        ----------
        scalars : str
            Name of the point or cell data array holding the sampled values.

        param_grid : dict[str, list]
            ``Krige`` parameters to search, for example
            ``{'variogram_model': ['linear', 'spherical']}``.

        cv : int, default: 5
            Number of cross-validation folds.

        scoring : str, optional
            scikit-learn scoring name. Defaults to the R² score.

        method : str, default: 'ordinary'
            Kriging method, ``'ordinary'`` or ``'universal'``.

        dim : int, optional
            Spatial dimension, 2 or 3. Detected from the samples by default.

        **kwargs : dict, optional
            Fixed options forwarded to :class:`pykrige.rk.Krige`.

        Returns
        -------
        CrossValidation
            Best parameters and score, all results, and the refitted model.

        """
        from pyvista_krige._advanced import cross_validate  # noqa: PLC0415

        return cross_validate(
            self._dataset,
            scalars,
            param_grid,
            cv=cv,
            scoring=scoring,
            method=method,
            dim=dim,
            **kwargs,
        )
