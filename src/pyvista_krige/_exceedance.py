"""Probability that a kriged field exceeds a threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from pyvista_krige._model import KrigingModel


def attributes(result: pv.DataSet, name: str) -> pv.DataSetAttributes:
    """Return the point data of ``result`` if it holds array ``name``, else its cell data."""
    if name in result.point_data or name not in result.cell_data:
        return result.point_data
    return result.cell_data


def uncertainty_name(result: pv.DataSet, scalars: str, variance: str | None) -> str:
    """Return the variance or std array name, defaulting to what ``predict`` writes."""
    if variance is not None:
        return variance
    for suffix in ('variance', 'std'):
        name = f'{scalars} {suffix}'
        if name in result.point_data or name in result.cell_data:
            return name
    msg = (
        f"No '{scalars} variance' or '{scalars} std' array found. Predict with variance "
        'enabled, or name the array with variance=.'
    )
    raise KeyError(msg)


def kriging_std(result: pv.DataSet, name: str, *, std: bool | None = None) -> np.ndarray:
    """Return the standard deviation from array ``name``, a variance unless it is a std.

    With ``std=None`` an array whose name ends in ``' std'`` is a standard
    deviation and any other is a variance.
    """
    values = np.asarray(attributes(result, name)[name], dtype=float)
    if std is None:
        std = name.endswith(' std')
    # Round-off can leave tiny negative variances at the samples.
    return values if std else np.sqrt(np.clip(values, 0, None))


def exceedance(
    result: pv.DataSet,
    scalars: str,
    threshold: float,
    *,
    variance: str | None = None,
    std: bool | None = None,
    below: bool = False,
    name: str | None = None,
    model: KrigingModel | None = None,
) -> pv.DataSet:
    """Return a copy of a kriging result with the probability of exceeding ``threshold``.

    Treats the estimate and its kriging variance as the mean and variance of a
    normal distribution at each point, the usual reading of a Gaussian kriging
    result. Where the variance is zero the probability is 0 or 1.

    For a model fitted with a ``transform``, pass it as ``model``. The
    probability is then computed in the transformed space, where the normal
    assumption holds, from the ``'<scalars> transformed'`` arrays that
    :meth:`KrigingModel.predict` writes. The transforms are increasing, so the
    value exceeds ``threshold`` exactly when its transform exceeds the
    transformed threshold.

    Parameters
    ----------
    result : pyvista.DataSet
        Output of :meth:`KrigingModel.predict` or ``dataset.krige.to_image``.

    scalars : str
        Name of the estimate array.

    threshold : float
        Value to compare against.

    variance : str, optional
        Name of the variance or standard deviation array. Defaults to
        ``'<scalars> variance'``, falling back to ``'<scalars> std'``.

    std : bool, optional
        Whether ``variance`` holds standard deviations. By default, names
        ending in ``' std'`` do.

    below : bool, default: False
        Give the probability of falling below ``threshold`` instead.

    name : str, optional
        Name of the output array. Defaults to ``'P(<scalars> > <threshold>)'``.

    model : KrigingModel, optional
        Model that predicted ``result``. Required when it has a ``transform``.

    Returns
    -------
    pyvista.DataSet
        Copy of ``result`` with the probability as point data, or as cell
        data when the estimate is cell data.

    Raises
    ------
    ValueError
        If ``result`` has transformed arrays but no transformed ``model`` is
        given, or if ``variance`` or ``std`` is given with one.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> import pyvista_krige
    >>> rng = np.random.default_rng(0)
    >>> samples = pv.PolyData(rng.uniform(0, 10, (30, 3)) * [1, 1, 0])
    >>> samples['height'] = samples.points[:, 0]
    >>> image = samples.krige.to_image('height', dimensions=(10, 10))
    >>> image = pyvista_krige.exceedance(image, 'height', 5.0)
    >>> image['P(height > 5)'].min() >= 0.0
    np.True_

    Skewed values are kriged as logarithms, so pass the model:

    >>> samples['grade'] = np.exp(samples.points[:, 0] / 5)
    >>> model = samples.krige.fit('grade', transform='log')
    >>> result = model.predict(image)
    >>> result = pyvista_krige.exceedance(result, 'grade', 2.0, model=model)
    >>> float(result['P(grade > 2)'].max()) > 0.99
    True

    """
    from scipy.stats import norm  # noqa: PLC0415

    transform = None if model is None else model.transform
    data = attributes(result, scalars)
    if transform is None and f'{scalars} transformed' in data:
        msg = (
            f"result holds '{scalars} transformed', so it was predicted by a transformed model. "
            'Pass that model as model= to compute the probability in the transformed space.'
        )
        raise ValueError(msg)
    if transform is not None:
        if variance is not None or std is not None:
            msg = 'variance and std cannot be set with a transformed model.'
            raise ValueError(msg)
        kriged = f'{scalars} transformed'
        mean = np.asarray(data[kriged], dtype=float)
        sigma = kriging_std(result, uncertainty_name(result, kriged, None))
        cutoff = float(transform.forward(threshold))
    else:
        mean = np.asarray(data[scalars], dtype=float)
        sigma = kriging_std(result, uncertainty_name(result, scalars, variance), std=std)
        cutoff = threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        probability = norm.sf((cutoff - mean) / np.where(sigma > 0, sigma, np.inf))
    probability = np.where(sigma > 0, probability, (mean > cutoff).astype(float))
    probability[~np.isfinite(mean)] = np.nan
    if below:
        probability = 1.0 - probability
    if name is None:
        name = f'P({scalars} {"<" if below else ">"} {threshold:g})'
    out = result.copy()
    attributes(out, scalars)[name] = probability
    return out
