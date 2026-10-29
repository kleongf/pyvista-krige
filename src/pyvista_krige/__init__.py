"""Kriging for PyVista datasets, powered by PyKrige."""

from __future__ import annotations

from pyvista_krige import _accessor  # noqa: F401  registers ``DataSet.krige``
from pyvista_krige import _component  # noqa: F401  registers ``Plotter.krige``
from pyvista_krige._accessor import KrigeAccessor
from pyvista_krige._advanced import CrossValidation
from pyvista_krige._component import KrigeComponent
from pyvista_krige._model import KrigingModel
from pyvista_krige._model import Statistics
from pyvista_krige._variogram import Variogram

__all__ = [
    'CrossValidation',
    'KrigeAccessor',
    'KrigeComponent',
    'KrigingModel',
    'Statistics',
    'Variogram',
]
