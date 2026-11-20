"""Register the ``krige`` accessor on :class:`pyvista.DataSet`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv

from pyvista_krige._targets import image_spanning

if TYPE_CHECKING:
    from pyvista_krige._advanced import CrossValidation
    from pyvista_krige._decluster import DeclusterScan
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
        transform: str | None = None,
        transform_limits: tuple[float | None, float | None] | None = None,
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

        transform : str, optional
            Krige transformed values, for skewed data such as grades,
            permeability or concentrations:

            - ``'log'``: the natural logarithm. Needs positive values.
            - ``'normal_score'``: standard normal scores of the empirical
              distribution.

            :meth:`KrigingModel.predict` back-transforms the estimate and
            variance to the original units, and
            :meth:`KrigingModel.cross_validate` reports errors in them.
            :attr:`KrigingModel.variogram` and :attr:`KrigingModel.statistics`
            stay in the transformed space. Pass the model to
            :func:`pyvista_krige.exceedance`.

        transform_limits : tuple[float | None, float | None], optional
            Smallest and largest possible values for ``transform='normal_score'``.
            Its distribution extends linearly from the smallest and largest
            samples to these. By default the outermost segments of the
            empirical distribution are extended, which stays within half a
            sample spacing of the sample range.

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

            Three options are handled here rather than by PyKrige:

            - ``detrend=True`` (universal kriging): fit the variogram to the
              least-squares residuals of the drift instead of the raw values.
              PyKrige fits the raw values, so a strong trend inflates the sill
              and range. Supports the ``regional_linear``, ``specified`` and
              ``functional`` drift terms.
            - ``max_lag``: fit the variogram only to pairs closer than this,
              since long lags have few pairs and often show the trend rather
              than the correlation. Half the sample extent is a common choice.
            - ``anisotropy='auto'``: estimate the anisotropy from directional
              variograms and pass it to PyKrige, instead of comparing the
              variograms by hand and setting the ``anisotropy_*`` options.
              The nugget and sill come from the omnidirectional variogram.
              The range is then fitted along eight horizontal directions to
              find the major axis and the ratio of the horizontal ranges, and
              along z for 3D samples. Read the result from
              :attr:`KrigingModel.anisotropy`. It needs a bounded
              ``variogram_model`` (``'gaussian'``, ``'spherical'``,
              ``'exponential'`` or ``'hole-effect'``, not PyKrige's default
              ``'linear'``) and cannot be combined with the ``anisotropy_*``
              options, a GSTools model or ``coordinates_type='geographic'``.
              The estimate uses the drift residuals with ``detrend=True``,
              ``nlags`` lag bins, and pairs up to ``max_lag``. Tilted
              anisotropy is not estimated, so ``anisotropy_angle_x`` and
              ``_y`` stay 0. The estimate is noisy with fewer than a few
              hundred samples. Combine it with ``max_lag``, which fits the
              variogram of the adjusted coordinates to short lags only.
              :meth:`KrigingModel.cross_validate` estimates the anisotropy
              again in every fold.

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

        The variation of ``height`` is along x only, so the correlation reaches
        further along y.

        >>> model = samples.krige.fit(
        ...     'height', variogram_model='spherical', anisotropy='auto'
        ... )
        >>> bool(model.anisotropy['anisotropy_scaling'] > 1)
        True

        """
        from pyvista_krige._model import KrigingModel  # noqa: PLC0415

        return KrigingModel.fit(
            self._dataset,
            scalars,
            method=method,
            dim=dim,
            transform=transform,
            transform_limits=transform_limits,
            **kwargs,
        )

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
        fitting. ``fit(..., anisotropy='auto')`` makes this comparison for you.
        Plot the result with ``pl.krige.add_variogram``.

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

    def decluster(
        self,
        cell_size: float | tuple[float, ...] | None = None,
        *,
        scalars: str | None = None,
        name: str = 'declustering weight',
        n_offsets: int = 10,
        n_sizes: int = 20,
        size_range: tuple[float, float] | None = None,
        aspect: tuple[float, ...] | None = None,
        maximize: bool = False,
    ) -> pv.DataSet:
        """Return a copy with cell declustering weights for clustered samples.

        Samples gathered where values are high or low, such as extra boreholes
        around a find, skew the mean and the histogram. Cell declustering lays a
        grid over the samples and gives each one the weight 1 over the number of
        samples in its cell. The weights are scaled to sum to the number of
        samples and averaged over shifted grid origins, so they do not depend on
        where the grid starts or on the order of the samples.

        The weights are written to ``point_data[name]``, or to ``cell_data`` when
        ``scalars`` is a cell array. The cell size used is stored in
        ``field_data['declustering cell size']``. Use the weights with
        :func:`numpy.average` or :func:`numpy.histogram` to get declustered
        statistics.

        Parameters
        ----------
        cell_size : float | tuple[float, ...], optional
            Cell size. A number gives square cells for flat samples and cubes for
            samples with a spread in z. A tuple ``(dx, dy)`` or ``(dx, dy, dz)``
            sets each axis, for example a thin dz for layered samples. By default
            the size is chosen by scanning, see :meth:`decluster_scan`, which
            needs ``scalars``.

        scalars : str, optional
            Name of the point or cell data array whose mean the scan declusters.
            Cell arrays are sampled at the cell centers. Only needed to choose
            the cell size.

        name : str, default: 'declustering weight'
            Name of the weight array.

        n_offsets : int, default: 10
            Number of shifted grid origins to average over.

        n_sizes : int, default: 20
            Number of cell sizes to scan.

        size_range : tuple[float, float], optional
            Smallest and largest cell size to scan. By default, from the 10th
            percentile of the nearest-neighbor distances to half the longest
            extent.

        aspect : tuple[float, ...], optional
            Shape of the scanned cell, so the cell is ``size * aspect``. Its
            length sets the dimension. By default cells are square or cubic.

        maximize : bool, default: False
            Choose the cell size with the highest declustered mean instead of the
            lowest. Use it when the samples are clustered where values are low.

        Returns
        -------
        pyvista.DataSet
            Copy of the dataset with the weights.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> regular = np.mgrid[0:10:5j, 0:10:5j].reshape(2, -1).T
        >>> cluster = rng.uniform(4, 5, (25, 2))
        >>> xy = np.vstack([regular, cluster])
        >>> samples = pv.PolyData(np.column_stack([xy, np.zeros(len(xy))]))
        >>> weighted = samples.krige.decluster(cell_size=2.0)
        >>> weights = weighted['declustering weight']
        >>> bool(weights[25:].mean() < weights[:25].mean())
        True

        """
        from pyvista_krige._decluster import decluster  # noqa: PLC0415

        scan = None
        if cell_size is None:
            scan = self.decluster_scan(
                scalars,  # type: ignore[arg-type]
                n_offsets=n_offsets,
                n_sizes=n_sizes,
                size_range=size_range,
                aspect=aspect,
                maximize=maximize,
            )
        return decluster(
            self._dataset,
            cell_size,
            scalars=scalars,
            name=name,
            n_offsets=n_offsets,
            scan=scan,
        )

    def decluster_scan(
        self,
        scalars: str,
        *,
        n_offsets: int = 10,
        n_sizes: int = 20,
        size_range: tuple[float, float] | None = None,
        aspect: tuple[float, ...] | None = None,
        maximize: bool = False,
    ) -> DeclusterScan:
        """Scan cell sizes for the one with the lowest declustered mean.

        For each of ``n_sizes`` cell sizes, spaced geometrically, this computes the
        declustered mean of ``scalars``. When the samples cluster where values are
        high, the mean drops as the cells grow to hold the clusters and rises
        again once the cells hold everything, so the lowest mean marks the best
        size. Plot ``cell_sizes[:, 0]`` against ``means`` to check that the
        minimum is a clear one. :meth:`decluster` runs this scan when it is not
        given a ``cell_size``.

        Parameters
        ----------
        scalars : str
            Name of the point or cell data array whose mean to decluster. Samples
            with NaN or infinite values do not count towards the mean.

        n_offsets : int, default: 10
            Number of shifted grid origins to average over.

        n_sizes : int, default: 20
            Number of cell sizes to scan.

        size_range : tuple[float, float], optional
            Smallest and largest cell size to scan. By default, from the 10th
            percentile of the nearest-neighbor distances to half the longest
            extent.

        aspect : tuple[float, ...], optional
            Shape of the scanned cell, so the cell is ``size * aspect``. Its
            length sets the dimension. By default cells are square for flat
            samples and cubic for samples with a spread in z.

        maximize : bool, default: False
            Choose the highest declustered mean instead of the lowest. Use it when
            the samples are clustered where values are low. A warning says when the
            best size is at the end of the range, which usually means the wrong
            choice of direction.

        Returns
        -------
        DeclusterScan
            Cell sizes, declustered means and the index of the chosen size.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> regular = np.mgrid[0:10:5j, 0:10:5j].reshape(2, -1).T
        >>> cluster = rng.uniform(4, 5, (25, 2))
        >>> xy = np.vstack([regular, cluster])
        >>> samples = pv.PolyData(np.column_stack([xy, np.zeros(len(xy))]))
        >>> samples['grade'] = np.r_[np.ones(25), np.full(25, 3.0)]
        >>> scan = samples.krige.decluster_scan('grade')
        >>> bool(scan.means[scan.best] < samples['grade'].mean())
        True

        """
        from pyvista_krige._decluster import scan_dataset  # noqa: PLC0415

        return scan_dataset(
            self._dataset,
            scalars,
            n_offsets=n_offsets,
            n_sizes=n_sizes,
            size_range=size_range,
            aspect=aspect,
            maximize=maximize,
        )
