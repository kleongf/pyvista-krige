"""Register the ``krige`` component on :class:`pyvista.Plotter`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista_krige._model import KrigingModel
    from pyvista_krige._variogram import ExperimentalVariogram
    from pyvista_krige._variogram import Variogram

# Series colors for charts with several variograms.
_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink')


@pv.register_plotter_component('krige')
class KrigeComponent:
    """Plotting helpers for kriging results.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter that owns this component.

    """

    def __init__(self, plotter: pv.Plotter) -> None:
        self._plotter = plotter

    def add_variogram(
        self,
        variogram: KrigingModel
        | Variogram
        | ExperimentalVariogram
        | Sequence[KrigingModel | Variogram | ExperimentalVariogram],
        *,
        loc: tuple[float, float] = (0.0, 0.0),
        size: tuple[float, float] = (1.0, 1.0),
        n_model_points: int = 100,
        labels: Sequence[str] | None = None,
    ) -> pv.Chart2D:
        """Add a chart of experimental variograms and fitted models.

        Parameters
        ----------
        variogram : KrigingModel | Variogram | ExperimentalVariogram | sequence
            Fitted model, its :attr:`KrigingModel.variogram`, an
            :class:`ExperimentalVariogram` from ``samples.krige.variogram``, or
            a list of these to compare, such as horizontal and vertical
            variograms. Experimental variograms are drawn as points only.

        loc : sequence[float], default: (0.0, 0.0)
            Lower-left corner of the chart, in normalized viewport coordinates.

        size : sequence[float], default: (1.0, 1.0)
            Width and height of the chart, in normalized viewport coordinates.

        n_model_points : int, default: 100
            Number of distances at which each model curve is evaluated.

        labels : sequence[str], optional
            Legend label of each variogram. Defaults to ``'Experimental'`` and
            the model name for a single model, otherwise the direction or model
            name.

        Returns
        -------
        pyvista.Chart2D
            The chart, already added to the active renderer.

        """
        import numpy as np  # noqa: PLC0415

        items = list(variogram) if isinstance(variogram, (list, tuple)) else [variogram]
        items = [getattr(item, 'variogram', item) for item in items]
        if labels is not None and len(labels) != len(items):
            msg = f'Got {len(labels)} labels for {len(items)} variograms.'
            raise ValueError(msg)
        chart = pv.Chart2D(size=size, loc=loc, x_label='Lag distance', y_label='Semivariance')
        distances = np.linspace(0.0, max(item.lags.max() for item in items), n_model_points)
        for k, item in enumerate(items):
            color = _COLORS[k % len(_COLORS)]
            fitted = hasattr(item, 'model')
            if labels is not None:
                label = labels[k]
            elif len(items) == 1:
                label = 'Experimental'
            else:
                label = item.model_name if fitted else item.label
            chart.scatter(item.lags, item.semivariance, color=color, label=label)
            if fitted:
                line_label = item.model_name if len(items) == 1 else ''
                line_color = 'r' if len(items) == 1 else color
                chart.line(
                    distances, item.model(distances), color=line_color, width=2.0, label=line_label
                )
        self._plotter.add_chart(chart)
        return chart

    def add_uncertainty(
        self,
        result: pv.DataSet,
        scalars: str,
        *,
        variance: str | None = None,
        min_opacity: float = 0.1,
        **kwargs: Any,
    ) -> pv.Actor:
        """Add a kriging result whose opacity fades where the variance is high.

        Parameters
        ----------
        result : pyvista.DataSet
            Output of :meth:`KrigingModel.predict` or ``dataset.krige.to_image``.

        scalars : str
            Name of the estimate array.

        variance : str, optional
            Name of the variance or standard deviation array. Defaults to
            ``'<scalars> variance'``, falling back to ``'<scalars> std'``.

        min_opacity : float, default: 0.1
            Opacity of the least certain point. Opacity falls linearly from 1
            at the lowest uncertainty to this value at the highest.

        **kwargs : dict, optional
            Options forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.Actor
            The actor of the added mesh.

        """
        if variance is None:
            variance = f'{scalars} variance'
            if variance not in result.point_data:
                variance = f'{scalars} std'
        if min_opacity >= 1.0:
            return self._plotter.add_mesh(result, scalars=scalars, **kwargs)

        import numpy as np  # noqa: PLC0415

        # Variances are in squared data units, but opacity must lie in [0, 1].
        uncertainty = np.asarray(result.point_data[variance], dtype=float)
        low, high = np.nanmin(uncertainty), np.nanmax(uncertainty)
        scaled = (uncertainty - low) / (high - low) if high > low else np.zeros_like(uncertainty)
        opacity = 1.0 - (1.0 - min_opacity) * np.nan_to_num(scaled, nan=1.0)
        return self._plotter.add_mesh(result, scalars=scalars, opacity=opacity, **kwargs)
