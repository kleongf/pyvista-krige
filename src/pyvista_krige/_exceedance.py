"""Probability that a kriged field exceeds a threshold."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def uncertainty_name(result: pv.DataSet, scalars: str, variance: str | None) -> str:
    """Return the variance or std array name, defaulting to what ``predict`` writes."""
    if variance is not None:
        return variance
    for suffix in ('variance', 'std'):
        name = f'{scalars} {suffix}'
        if name in result.point_data:
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
    values = np.asarray(result.point_data[name], dtype=float)
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
) -> pv.DataSet:
    """Return a copy of a kriging result with the probability of exceeding ``threshold``.

    Treats the estimate and its kriging variance as the mean and variance of a
    normal distribution at each point, the usual reading of a Gaussian kriging
    result. Where the variance is zero the probability is 0 or 1.

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

    Returns
    -------
    pyvista.DataSet
        Copy of ``result`` with the probability as point data.

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

    """
    from scipy.stats import norm  # noqa: PLC0415

    mean = np.asarray(result.point_data[scalars], dtype=float)
    sigma = kriging_std(result, uncertainty_name(result, scalars, variance), std=std)
    with np.errstate(divide='ignore', invalid='ignore'):
        probability = norm.sf((threshold - mean) / np.where(sigma > 0, sigma, np.inf))
    probability = np.where(sigma > 0, probability, (mean > threshold).astype(float))
    probability[~np.isfinite(mean)] = np.nan
    if below:
        probability = 1.0 - probability
    if name is None:
        name = f'P({scalars} {"<" if below else ">"} {threshold:g})'
    out = result.copy()
    out.point_data[name] = probability
    return out
