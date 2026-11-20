"""Translate PyVista-style options for PyKrige's advanced models."""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

import numpy as np
import pyvista as pv

from pyvista_krige._targets import grid_axes
from pyvista_krige._targets import sample_arrays

if TYPE_CHECKING:
    from pyvista_krige._model import KrigingModel
    from pyvista_krige._targets import Samples


def translate_drift(samples: Samples, kwargs: dict[str, Any]) -> list[str] | None:
    """Convert dataset-based drift options in ``kwargs`` to PyKrige arrays, in place.

    ``specified_drift`` names arrays on the samples, which ``predict`` later reads
    from the target; the names are returned. ``external_drift`` may be a
    single-layer ``ImageData`` with the drift in ``external_drift_scalars``.
    """
    drift_names = kwargs.get('specified_drift')
    if drift_names is not None:
        if isinstance(drift_names, str):
            drift_names = [drift_names]
        if not all(isinstance(name, str) for name in drift_names):
            msg = (
                'specified_drift must name arrays present on both the samples and the '
                'prediction target.'
            )
            raise TypeError(msg)
        kwargs['specified_drift'] = [samples.aligned(name) for name in drift_names]

    image = kwargs.get('external_drift')
    if isinstance(image, pv.DataSet):
        axes = grid_axes(image, dim=2) if isinstance(image, pv.ImageData) else None
        if axes is None:
            msg = 'external_drift must be a single-layer, axis-aligned ImageData.'
            raise ValueError(msg)
        name = kwargs.pop('external_drift_scalars', None) or image.point_data.active_scalars_name
        x, y = axes
        kwargs['external_drift'] = np.asarray(image.point_data[name], dtype=float).reshape(
            len(y), len(x)
        )
        kwargs['external_drift_x'] = x
        kwargs['external_drift_y'] = y
    return drift_names


# Drift options that PyKrige ignores unless their term is also in ``drift_terms``.
_IMPLIED_TERMS = {
    'specified_drift': 'specified',
    'functional_drift': 'functional',
    'external_drift': 'external_Z',
    'point_drift': 'point_log',
}


def imply_drift_terms(kwargs: dict[str, Any]) -> None:
    """Add the ``drift_terms`` entry for each drift option given in ``kwargs``, in place.

    PyKrige silently ignores ``specified_drift`` and the like unless the matching
    term is also listed in ``drift_terms``.
    """
    terms = list(kwargs.get('drift_terms') or [])
    for option, term in _IMPLIED_TERMS.items():
        if kwargs.get(option) is not None and term not in terms:
            terms.append(term)
    if terms:
        kwargs['drift_terms'] = terms


def drift_matrix(coords: np.ndarray, kwargs: dict[str, Any]) -> np.ndarray:
    """Return the drift design matrix at ``coords``, with a leading column of ones.

    ``coords`` has one column per spatial dimension and should be the coordinates
    PyKrige evaluates the drift on. ``kwargs`` are translated PyKrige options, with
    ``specified_drift`` already holding arrays.
    """
    columns = [np.ones(len(coords))]
    for term in kwargs.get('drift_terms') or []:
        if term == 'regional_linear':
            columns.extend(coords.T)
        elif term == 'specified':
            columns.extend(np.asarray(a, dtype=float) for a in kwargs['specified_drift'])
        elif term == 'functional':
            columns.extend(
                np.broadcast_to(np.asarray(f(*coords.T), dtype=float), len(coords))
                for f in kwargs['functional_drift']
            )
        else:
            msg = (
                f'detrend supports the regional_linear, specified and functional drift '
                f'terms, not {term!r}.'
            )
            raise ValueError(msg)
    if len(columns) == 1:
        msg = 'detrend needs at least one drift term, such as drift_terms=["regional_linear"].'
        raise ValueError(msg)
    return np.column_stack(columns)


def detrended(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Return the least-squares residuals of ``values`` on the columns of ``design``."""
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def resolve_dim(points: np.ndarray, kwargs: dict[str, Any], dim: int | None, method: str) -> int:
    """Choose the spatial dimension and reject option combinations PyKrige cannot fit.

    Without an explicit ``dim``, a GSTools ``CovModel`` decides; otherwise samples
    with a spread in z are 3D.
    """
    model_dim = getattr(kwargs.get('variogram_model'), 'dim', None)
    if dim is None:
        dim = model_dim if model_dim in (2, 3) else (3 if np.ptp(points[:, 2]) > 0 else 2)
    elif dim not in (2, 3):
        msg = f'dim must be 2 or 3, got {dim!r}.'
        raise ValueError(msg)
    if model_dim is not None and model_dim != dim:
        msg = f'GSTools model has dim={model_dim} but dim={dim} was requested.'
        raise ValueError(msg)
    geographic = kwargs.get('coordinates_type', 'euclidean') == 'geographic'
    if geographic and (method != 'ordinary' or dim != 2):
        msg = "coordinates_type='geographic' works only with 2D ordinary kriging."
        raise ValueError(msg)
    return dim


def _require_sklearn() -> None:
    """Raise a helpful error when scikit-learn is missing."""
    try:
        import sklearn  # noqa: F401, PLC0415
    except ImportError:
        msg = 'This method needs scikit-learn: pip install "pyvista-krige[sklearn]".'
        raise ImportError(msg) from None


def _sklearn_method(method: str, dim: int) -> str:
    """Return PyKrige's scikit-learn method name, such as ``'ordinary3d'``."""
    return f'{method}3d' if dim == 3 else method


def predictor_matrix(dataset: pv.DataSet, predictors: list[str]) -> np.ndarray:
    """Stack the named point data arrays of ``dataset`` into an ``(n_points, n)`` matrix."""
    return np.column_stack([np.asarray(dataset.point_data[name]) for name in predictors])


def regression_kriging(
    samples: pv.DataSet,
    target: pv.DataSet,
    scalars: str,
    *,
    predictors: list[str],
    model: object | None,
    method: str,
    dim: int | None,
    classify: bool,
    **kwargs: Any,
) -> pv.DataSet:
    """Fit a scikit-learn model to ``predictors``, krige its residuals, and predict."""
    _require_sklearn()
    if classify:
        from pykrige.ck import ClassificationKriging as Kriging  # noqa: PLC0415

        model_key = 'classification_model'
    else:
        from pykrige.rk import RegressionKriging as Kriging  # noqa: PLC0415

        model_key = 'regression_model'
    if model is not None:
        kwargs[model_key] = model
    found = sample_arrays(samples, scalars)
    points = found.points
    dim = resolve_dim(points, kwargs, dim, method)
    kriging = Kriging(method=_sklearn_method(method, dim), **kwargs)
    values = found.values
    if classify:
        # ClassificationKriging one-hot encodes the labels, which needs a column vector.
        values = values.astype(int).reshape(-1, 1)
    features = np.column_stack([found.aligned(name) for name in predictors])
    # PyKrige reports progress with print(); keep it off the user's stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        kriging.fit(features, points[:, :dim], values)
        estimate = kriging.predict(predictor_matrix(target, predictors), target.points[:, :dim])
    out = target.copy()
    out.point_data[scalars] = estimate
    return out


class CrossValidation(NamedTuple):
    """Result of :meth:`KrigeAccessor.cross_validate`.

    Attributes
    ----------
    best_params : dict
        Parameter combination with the best mean test score.

    best_score : float
        Mean cross-validated score of ``best_params``.

    results : dict
        scikit-learn's ``cv_results_`` for every combination.

    model : KrigingModel
        Model refitted to all samples with ``best_params``.

    """

    best_params: dict[str, Any]
    best_score: float
    results: dict[str, Any]
    model: KrigingModel


# Krige options that only affect prediction, so they are not fit options.
_PREDICT_ONLY = ('n_closest_points',)


def cross_validate(
    samples: pv.DataSet,
    scalars: str,
    param_grid: dict[str, list[Any]],
    *,
    cv: int,
    scoring: str | None,
    method: str,
    dim: int | None,
    **kwargs: Any,
) -> CrossValidation:
    """Grid-search kriging parameters with scikit-learn and refit the best model."""
    _require_sklearn()
    from pykrige.rk import Krige  # noqa: PLC0415
    from sklearn.model_selection import GridSearchCV  # noqa: PLC0415

    from pyvista_krige._model import KrigingModel  # noqa: PLC0415

    dataset_options = {
        'specified_drift',
        'external_drift',
        'detrend',
        'max_lag',
        'anisotropy',
    } & set(kwargs)
    if dataset_options:
        msg = (
            f'cross_validate cannot search with {sorted(dataset_options)}. Fit the candidate '
            'models with samples.krige.fit(...) and compare model.cross_validate() instead.'
        )
        raise ValueError(msg)
    found = sample_arrays(samples, scalars)
    dim = resolve_dim(found.points, kwargs, dim, method)
    search = GridSearchCV(
        Krige(method=_sklearn_method(method, dim), **kwargs), param_grid, cv=cv, scoring=scoring
    )
    search.fit(found.points[:, :dim], found.values)

    best = dict(search.best_params_)
    fit_kwargs = {**kwargs, **best}
    best_method = fit_kwargs.pop('method', method)
    if best_method.endswith('3d'):
        best_method, dim = best_method[:-2], 3
    for key in _PREDICT_ONLY:
        fit_kwargs.pop(key, None)
    model = KrigingModel.fit(samples, scalars, method=best_method, dim=dim, **fit_kwargs)
    return CrossValidation(best, float(search.best_score_), dict(search.cv_results_), model)
