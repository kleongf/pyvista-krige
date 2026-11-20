"""Pick the next sample locations by greedy kriging variance."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_krige._targets import resolve_where
from pyvista_krige._variogram import parameter_dict

if TYPE_CHECKING:
    from pyvista_krige._model import KrigingModel

# Fit options that only matter for fitting or reporting; the variogram is held fixed here.
_FIT_ONLY = ('detrend', 'max_lag', 'enable_statistics', 'enable_plotting', 'verbose')


def pick_samples(
    model: KrigingModel,
    target: pv.DataSet,
    n: int,
    *,
    where: str | np.ndarray | None,
    min_spacing: float | None,
) -> pv.PolyData:
    """Return the ``n`` points of ``target`` picked one at a time by largest kriging std.

    See :meth:`KrigingModel.suggest_samples`, which is the usual way to call this.
    """
    if isinstance(target, pv.MultiBlock):
        msg = 'MultiBlock targets are not supported. Use target.combine() first.'
        raise TypeError(msg)
    if n < 1:
        msg = f'n must be at least 1, got {n}.'
        raise ValueError(msg)
    if min_spacing is not None and min_spacing < 0:
        msg = f'min_spacing must not be negative, got {min_spacing}.'
        raise ValueError(msg)
    alive = resolve_where(target, where)
    alive = np.ones(target.n_points, dtype=bool) if alive is None else alive.copy()
    if n > alive.sum():
        msg = f'Asked for {n} picks but only {alive.sum()} candidate points are available.'
        raise ValueError(msg)

    scalars, dim = model._scalars, model.dim
    drift_names = model._drift_names or []
    base = model.samples
    pseudo_points = [base.points]
    pseudo = {scalars: [base[scalars]], **{name: [base[name]] for name in drift_names}}
    fit_kwargs = _fixed_variogram_kwargs(model)

    picked: list[int] = []
    stds: list[float] = []
    current = model
    for step in range(n):
        # Predicting on the target keeps PyKrige's fast grid path, masked to the candidates.
        out = current.predict(target, where=alive)
        variance = np.asarray(out[f'{scalars} variance'], dtype=float)
        variance = np.where(alive & np.isfinite(variance), variance, -np.inf)
        pick = int(np.argmax(variance))
        if variance[pick] == -np.inf:
            msg = 'The kriging variance is not finite at any candidate point.'
            raise ValueError(msg)
        picked.append(pick)
        stds.append(float(np.sqrt(max(variance[pick], 0.0))))
        alive[pick] = False
        if min_spacing is not None:
            gap = np.linalg.norm(target.points[:, :dim] - target.points[pick, :dim], axis=1)
            alive &= gap >= min_spacing
        if step == n - 1:
            break
        if not alive.any():
            msg = (
                f'Only {len(picked)} of {n} picks fit with min_spacing={min_spacing}; '
                'no candidate points are left.'
            )
            warnings.warn(msg, stacklevel=3)
            break
        # Kriging variance does not depend on the values, so the estimate stands in.
        estimate = float(out[scalars][pick])
        pseudo_points.append(target.points[[pick]])
        pseudo[scalars].append(
            np.array([estimate if np.isfinite(estimate) else base[scalars].mean()])
        )
        for name in drift_names:
            pseudo[name].append(np.asarray(target.point_data[name], dtype=float)[[pick]])
        cloud = pv.PolyData(np.vstack(pseudo_points))
        for name, parts in pseudo.items():
            cloud.point_data[name] = np.concatenate(parts)
        current = type(model).fit(cloud, scalars, method=model._method, dim=dim, **fit_kwargs)

    picks = pv.PolyData(target.points[picked])
    picks['pick order'] = np.arange(1, len(picked) + 1)
    picks['std before pick'] = np.array(stds)
    return picks


def _fixed_variogram_kwargs(model: KrigingModel) -> dict[str, Any]:
    """Return fit options that rebuild ``model`` on more samples with its variogram held fixed.

    A named variogram passes its fitted parameters to PyKrige, which then fits nothing.
    A custom or GSTools variogram is fixed already, so its fit options carry over.
    """
    instance = model.instance
    kwargs = {k: v for k, v in model._fit_kwargs.items() if k not in _FIT_ONLY}
    name = instance.variogram_model
    if name != 'custom':
        kwargs['variogram_model'] = name
        kwargs['variogram_parameters'] = parameter_dict(name, instance.variogram_model_parameters)
    return kwargs
