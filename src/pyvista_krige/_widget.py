"""Sliders that retune a fitted variogram and re-krige a preview as they move."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista_krige._model import KrigingModel

# Warn above this many preview points: every slider move krige all of them.
MAX_PREVIEW_POINTS = 50_000

# Slider titles in PyKrige's internal parameter order.
_TITLES = {
    'linear': ('Slope', 'Nugget'),
    'power': ('Scale', 'Exponent', 'Nugget'),
    'spherical': ('Partial sill', 'Range', 'Nugget'),
    'exponential': ('Partial sill', 'Range', 'Nugget'),
    'gaussian': ('Partial sill', 'Range', 'Nugget'),
    'hole-effect': ('Partial sill', 'Range', 'Nugget'),
}

# Sliders stack at the lower left of the viewport, the first parameter on top.
_SLIDER_X = (0.03, 0.33)
_SLIDER_Y0 = 0.09
_SLIDER_DY = 0.11


@dataclass(frozen=True)
class _Slider:
    """One slider: the parameter it sets, its title and its limits."""

    index: int
    title: str
    low: float
    high: float


def slider_specs(model: KrigingModel) -> list[_Slider]:
    """Return the sliders that tune ``model``'s variogram, in parameter order.

    Raises
    ------
    ValueError
        If the variogram is custom or from GSTools, so its parameters are unknown.

    """
    vario = model.variogram
    titles = _TITLES.get(vario.model_name)
    if titles is None:
        msg = (
            f'Cannot tune a {vario.model_name!r} variogram. Fit a linear, power, spherical, '
            'exponential, gaussian or hole-effect model.'
        )
        raise ValueError(msg)
    values = vario.parameters
    max_lag = float(vario.lags.max())
    # Typical semivariance, to size limits when a fitted value is 0. PyKrige fits
    # nuggets of about 1e-16 and puts the range on the largest lag, so limits cannot
    # be plain multiples of the fitted values.
    scale = float(np.nanmax(vario.semivariance))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    name = vario.model_name
    specs = []
    for index, title in enumerate(titles):
        value = float(values[index])
        if title == 'Nugget':
            low, high = 0.0, max(2 * value, 0.5 * scale)
        elif title == 'Range':
            low, high = float(vario.lags.min()), max(2 * value, max_lag)
        elif title == 'Exponent':
            low, high = 0.05, 1.95  # PyKrige's power model needs 0 < exponent < 2
        else:
            # A partial sill, slope or scale of 0 makes the kriging matrix singular.
            fallback = scale / max_lag if name in ('linear', 'power') else scale
            high = 2 * value if value > 0 else fallback
            low = 0.01 * high
        specs.append(_Slider(index, title, low, high))
    return specs


def _pykrige_parameters(name: str, values: list[float]) -> list[float]:
    """Return the list PyKrige reads as ``variogram_parameters`` for these values."""
    if name in ('linear', 'power'):
        return list(values)
    # PyKrige reads the first value of a list as the total sill and subtracts the
    # nugget, so passing the partial sill unchanged would shrink it on every move.
    psill, rng, nugget = values
    return [psill + nugget, rng, nugget]


class VariogramWidget:
    """Own the sliders, the preview mesh and the chart of one variogram widget."""

    def __init__(
        self,
        plotter: pv.Plotter,
        model: KrigingModel,
        preview: pv.DataSet,
        *,
        callback: Callable[[KrigingModel], object] | None,
        interaction_event: str,
        scalars: str | None,
        show_variogram: bool,
        chart_loc: tuple[float, float],
        chart_size: tuple[float, float],
        mesh_kwargs: dict[str, Any],
    ) -> None:
        if preview.n_points > MAX_PREVIEW_POINTS:
            msg = (
                f'The preview has {preview.n_points} points and is kriged on every slider '
                f'move. Pass a coarse preview of at most {MAX_PREVIEW_POINTS} points to '
                'keep the sliders responsive.'
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        self.plotter = plotter
        self.base = model
        self.model = model
        self.preview = preview
        self.callback = callback
        self.scalars = model._scalars
        self.name = model.variogram.model_name
        self.specs = slider_specs(model)
        self.values = [float(v) for v in model.variogram.parameters]

        result = model.predict(preview)
        self.mesh = result
        self.color_by = scalars or self.scalars
        if self.color_by not in result.array_names:
            msg = f'The preview has no array {self.color_by!r}. Choose from {result.array_names}.'
            raise ValueError(msg)
        # The estimate keeps its first range so changes show up as changes in color. Other
        # arrays, such as the variance, change by orders of magnitude and are rescaled.
        self.rescale = self.color_by != self.scalars and 'clim' not in mesh_kwargs
        if self.color_by == self.scalars:
            values = result[self.color_by]
            mesh_kwargs.setdefault('clim', (float(np.nanmin(values)), float(np.nanmax(values))))
        # Keep the color bar clear of the sliders and the chart.
        mesh_kwargs.setdefault(
            'scalar_bar_args',
            {'position_x': 0.3, 'position_y': 0.9, 'width': 0.4, 'height': 0.06},
        )
        self.actor = plotter.add_mesh(result, scalars=self.color_by, **mesh_kwargs)

        self.line = None
        if show_variogram:
            chart = plotter.krige.add_variogram(model, loc=chart_loc, size=chart_size)
            (self.line,) = chart.plots('line')
            chart.background_color = plotter.background_color  # hide the mesh behind it
            self.distances = np.asarray(self.line.x, dtype=float)
            # Keep the axes still so the curve is judged against fixed experimental points.
            top = 1.1 * max(float(np.nanmax(model.variogram.semivariance)), self.curve().max())
            chart.y_range = (0.0, top)
            chart.x_range = (0.0, float(self.distances.max()))

        for k, spec in enumerate(self.specs):
            y = _SLIDER_Y0 + (len(self.specs) - 1 - k) * _SLIDER_DY
            plotter.add_slider_widget(
                lambda value, spec=spec: self.change(spec.index, value),
                (spec.low, spec.high),
                value=min(max(self.values[spec.index], spec.low), spec.high),
                title=spec.title,
                pointa=(_SLIDER_X[0], y),
                pointb=(_SLIDER_X[1], y),
                interaction_event=interaction_event,
                style='modern',
                fmt='%.3g',
                title_height=0.02,
            )

    def curve(self) -> np.ndarray:
        """Return the current model evaluated on the chart's distances."""
        return self.model.variogram.model(self.distances)

    def change(self, index: int, value: float) -> None:
        """Set parameter ``index`` to ``value``, then rebuild the model and the preview."""
        if value == self.values[index]:
            return  # sliders report their start value when created
        values = list(self.values)
        values[index] = float(value)
        try:
            model = self.base.with_variogram(
                self.name, variogram_parameters=_pykrige_parameters(self.name, values)
            )
            result = model.predict(self.preview)
        except np.linalg.LinAlgError:
            msg = (
                'The kriging matrix is singular for these variogram parameters. '
                'Keeping the previous model.'
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            return
        self.values = values
        self.model = model
        for name in result.array_names:
            self.mesh.point_data[name] = result.point_data[name]
        self.mesh.Modified()
        if self.rescale:
            values = self.mesh[self.color_by]
            self.actor.mapper.scalar_range = (float(np.nanmin(values)), float(np.nanmax(values)))
        if self.line is not None:
            self.line.update(self.distances, self.curve())
        self.plotter.render()
        if self.callback is not None:
            self.callback(model)
