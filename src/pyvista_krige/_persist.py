"""Save a fitted kriging model to ``.npz`` and load it back without refitting.

The file holds plain arrays and a JSON metadata string, so loading it never
runs code. Loading rebuilds the PyKrige object from the saved samples and the
fitted variogram parameters, which gives the same predictions bit for bit.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_krige._advanced import imply_drift_terms
from pyvista_krige._advanced import translate_drift
from pyvista_krige._targets import sample_arrays
from pyvista_krige._transform import LogTransform
from pyvista_krige._transform import NormalScoreTransform
from pyvista_krige._variogram import _PARAMETER_NAMES
from pyvista_krige._variogram import ExperimentalVariogram
from pyvista_krige._variogram import parameter_dict

if TYPE_CHECKING:
    import os

    from pyvista_krige._model import KrigingModel

FORMAT = 'pyvista-krige model'
VERSION = 1

# Fit options that ``load`` replaces with what the fit produced, so it never refits.
_REFIT_OPTIONS = (
    'anisotropy',
    'detrend',
    'max_lag',
    'transform',
    'transform_limits',
    'variogram_model',
    'variogram_parameters',
    'variogram_function',
    'enable_plotting',
)


def save_model(model: KrigingModel, path: str | os.PathLike) -> None:
    """Write ``model`` to ``path`` as ``.npz`` arrays plus JSON metadata."""
    instance = model.instance
    if instance.variogram_model not in _PARAMETER_NAMES:
        msg = (
            'Only models with a named PyKrige variogram can be saved, not a custom '
            'function or a GSTools CovModel.'
        )
        raise TypeError(msg)
    options = dict(model._fit_kwargs)
    if options.get('functional_drift') is not None:
        msg = 'A model with functional_drift cannot be saved: functions are not stored.'
        raise TypeError(msg)
    transform = options.pop('transform', None)
    transform_limits = options.pop('transform_limits', None)
    if isinstance(options.get('external_drift'), pv.DataSet):
        # Store the drift grid as the arrays PyKrige takes; refits read them the same way.
        external = {'external_drift': options.pop('external_drift')}
        if 'external_drift_scalars' in options:
            external['external_drift_scalars'] = options.pop('external_drift_scalars')
        translate_drift(None, external)
        options.update(external)
    arrays: dict[str, np.ndarray] = {}
    json_options = {}
    for key, value in options.items():
        if isinstance(value, np.ndarray):
            arrays[f'option {key}'] = value
        else:
            json_options[key] = value.item() if isinstance(value, np.generic) else value
    samples = model.samples
    drift_names = model._drift_names
    arrays['points'] = np.asarray(samples.points)
    arrays['values'] = np.asarray(samples.point_data[model._scalars])
    for i, name in enumerate(drift_names or []):
        arrays[f'drift {i}'] = np.asarray(samples.point_data[name])
    fitted = model.transform
    if isinstance(fitted, NormalScoreTransform):
        arrays['transform values'] = fitted.values
        arrays['transform cdf'] = fitted.cdf
    experimental = model._experimental
    if experimental is not None:
        arrays['experimental lags'] = experimental.lags
        arrays['experimental semivariance'] = experimental.semivariance
        arrays['experimental counts'] = experimental.counts
    metadata = {
        'format': FORMAT,
        'version': VERSION,
        'scalars': model._scalars,
        'method': model._method,
        'dim': model.dim,
        'drift_names': drift_names,
        'options': json_options,
        'variogram_model': instance.variogram_model,
        'variogram_parameters': [float(p) for p in instance.variogram_model_parameters],
        'anisotropy': model.anisotropy,
        'transform': None if fitted is None else fitted.name,
        'transform_option': transform,
        'transform_limits': transform_limits,
        'experimental_max_lag': None if experimental is None else experimental.max_lag,
    }
    # Raises TypeError for an option that is neither a JSON value nor an array.
    arrays['metadata'] = np.array(json.dumps(metadata))
    # An open file keeps numpy from appending '.npz' to a path without it.
    with open(path, 'wb') as file:
        np.savez_compressed(file, **arrays)


def load_model(path: str | os.PathLike) -> KrigingModel:
    """Read a model written by :func:`save_model`."""
    from pyvista_krige._model import KrigingModel  # noqa: PLC0415
    from pyvista_krige._model import _kriging_class  # noqa: PLC0415

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = _read_metadata(arrays, path)
    scalars, method, dim = metadata['scalars'], metadata['method'], metadata['dim']
    drift_names = metadata['drift_names']
    samples = pv.PolyData(arrays['points'])
    samples.point_data[scalars] = arrays['values']
    for i, name in enumerate(drift_names or []):
        samples.point_data[name] = arrays[f'drift {i}']
    options = dict(metadata['options'])
    for key, value in arrays.items():
        if key.startswith('option '):
            options[key.removeprefix('option ')] = value
    transform = None
    if metadata['transform'] == LogTransform.name:
        transform = LogTransform()
    elif metadata['transform'] == NormalScoreTransform.name:
        transform = NormalScoreTransform(arrays['transform values'], arrays['transform cdf'])

    model_name = metadata['variogram_model']
    kwargs = {k: v for k, v in options.items() if k not in _REFIT_OPTIONS}
    kwargs.update(metadata['anisotropy'])
    kwargs['variogram_model'] = model_name
    kwargs['variogram_parameters'] = parameter_dict(model_name, metadata['variogram_parameters'])
    found = sample_arrays(samples, scalars)
    translate_drift(found, kwargs)
    if method == 'universal':
        imply_drift_terms(kwargs)
    values = found.values if transform is None else transform.forward(found.values)
    instance = _kriging_class(method, dim)(*found.points.T[:dim], values, **kwargs)

    experimental = None
    if metadata['experimental_max_lag'] is not None:
        experimental = ExperimentalVariogram(
            lags=arrays['experimental lags'],
            semivariance=arrays['experimental semivariance'],
            counts=arrays['experimental counts'],
            direction=None,
            max_lag=metadata['experimental_max_lag'],
        )
    if metadata['transform_option'] is not None:
        options.update(
            transform=metadata['transform_option'], transform_limits=metadata['transform_limits']
        )
    return KrigingModel(
        instance,
        scalars,
        method=method,
        dim=dim,
        n_samples=len(values),
        drift_names=drift_names,
        samples=samples,
        fit_kwargs=options,
        experimental=experimental,
        transform=transform,
    )


def _read_metadata(arrays: dict[str, np.ndarray], path: Any) -> dict[str, Any]:
    """Return the metadata of a saved model, checking its format and version."""
    try:
        metadata = json.loads(str(arrays['metadata']))
    except (KeyError, ValueError):
        metadata = None
    if not isinstance(metadata, dict) or metadata.get('format') != FORMAT:
        msg = f'{path} is not a pyvista-krige model file.'
        raise ValueError(msg)
    version = metadata.get('version')
    if version != VERSION:
        msg = (
            f'{path} has model format version {version!r}, but this pyvista-krige reads '
            f'only version {VERSION}. A newer version needs a newer pyvista-krige.'
        )
        raise ValueError(msg)
    return metadata
