# Type checkers do not run the registration decorators, so declare the
# attributes they install at runtime.
from pyvista import BasePlotter
from pyvista import DataSet

from pyvista_krige._accessor import KrigeAccessor as KrigeAccessor
from pyvista_krige._advanced import CrossValidation as CrossValidation
from pyvista_krige._component import KrigeComponent as KrigeComponent
from pyvista_krige._model import KrigingModel as KrigingModel
from pyvista_krige._model import Statistics as Statistics
from pyvista_krige._variogram import Variogram as Variogram

DataSet.krige: KrigeAccessor  # type: ignore[attr-defined]
BasePlotter.krige: KrigeComponent  # type: ignore[attr-defined]

__all__ = [
    'CrossValidation',
    'KrigeAccessor',
    'KrigeComponent',
    'KrigingModel',
    'Statistics',
    'Variogram',
]
