"""Register the ``krige`` component on :class:`pyvista.Plotter`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista_krige._model import HoldOut
    from pyvista_krige._model import KrigingModel
    from pyvista_krige._variogram import ExperimentalVariogram
    from pyvista_krige._variogram import Variogram

# Series colors for charts with several variograms.
_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink')

# Variogram models whose second parameter is a range.
_BOUNDED = ('spherical', 'exponential', 'gaussian', 'hole-effect')


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
        from pyvista_krige._exceedance import uncertainty_name  # noqa: PLC0415

        variance = uncertainty_name(result, scalars, variance)
        if min_opacity >= 1.0:
            return self._plotter.add_mesh(result, scalars=scalars, **kwargs)

        import numpy as np  # noqa: PLC0415

        # Variances are in squared data units, but opacity must lie in [0, 1].
        uncertainty = np.asarray(result.point_data[variance], dtype=float)
        low, high = np.nanmin(uncertainty), np.nanmax(uncertainty)
        scaled = (uncertainty - low) / (high - low) if high > low else np.zeros_like(uncertainty)
        opacity = 1.0 - (1.0 - min_opacity) * np.nan_to_num(scaled, nan=1.0)
        return self._plotter.add_mesh(result, scalars=scalars, opacity=opacity, **kwargs)

    def add_exceedance(
        self,
        result: pv.DataSet,
        scalars: str,
        threshold: float,
        *,
        probability: float = 0.5,
        variance: str | None = None,
        below: bool = False,
        model: KrigingModel | None = None,
        **kwargs: Any,
    ) -> pv.Actor:
        """Add the surface where the field exceeds ``threshold`` with a given probability.

        With ``probability=0.9`` everything inside the surface is above
        ``threshold`` with at least 90 % probability, which is narrower than
        the plain contour at ``threshold`` wherever the kriging is uncertain.
        See :func:`pyvista_krige.exceedance`.

        Parameters
        ----------
        result : pyvista.DataSet
            Output of :meth:`KrigingModel.predict` or ``dataset.krige.to_image``.

        scalars : str
            Name of the estimate array.

        threshold : float
            Value to compare against.

        probability : float, default: 0.5
            Probability level to contour. ``0.5`` gives the contour of the
            estimate at ``threshold``.

        variance : str, optional
            Name of the variance or standard deviation array.

        below : bool, default: False
            Contour the probability of falling below ``threshold`` instead.

        model : KrigingModel, optional
            Model that predicted ``result``. Required when it has a ``transform``.

        **kwargs : dict, optional
            Options forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.Actor
            The actor of the probability surface.

        """
        from pyvista_krige._exceedance import exceedance  # noqa: PLC0415

        name = f'P({scalars} {"<" if below else ">"} {threshold:g})'
        prob = exceedance(
            result, scalars, threshold, variance=variance, below=below, name=name, model=model
        )
        surface = prob.contour([probability], scalars=name)
        kwargs.setdefault('color', 'tomato')
        kwargs.setdefault('label', f'{name} = {probability:g}')
        return self._plotter.add_mesh(surface, **kwargs)

    def add_residuals(
        self,
        holdout: HoldOut,
        *,
        cmap: str = 'RdBu_r',
        radius: float | None = None,
        max_z_score: float = 3.0,
        **kwargs: Any,
    ) -> pv.Actor:
        """Add cross-validation errors as spheres at the sample locations.

        Spheres are colored by the hold-out error (estimate minus observed) on
        a symmetric diverging map and sized by the absolute z-score, so
        samples the model explains badly stand out.

        Parameters
        ----------
        holdout : HoldOut
            Result of :meth:`KrigingModel.cross_validate`.

        cmap : str, default: 'RdBu_r'
            Diverging colormap for the errors.

        radius : float, optional
            Radius of a sphere with a z-score of ``max_z_score``. Defaults to
            2 % of the samples' bounding-box diagonal.

        max_z_score : float, default: 3.0
            Absolute z-score at which spheres stop growing.

        **kwargs : dict, optional
            Options forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns
        -------
        pyvista.Actor
            The actor of the spheres.

        """
        import numpy as np  # noqa: PLC0415

        samples = holdout.samples.copy()
        error_name = f'{holdout.scalars} error'
        z_score = np.abs(np.nan_to_num(samples[f'{holdout.scalars} z-score'], nan=0.0))
        if radius is None:
            radius = 0.02 * samples.length
        # Spheres shrink to a third of full size at z = 0.
        samples['_size'] = radius * (1 + 2 * np.clip(z_score, 0, max_z_score) / max_z_score) / 3
        glyphs = samples.glyph(scale='_size', orient=False, factor=2.0, geom=pv.Sphere(radius=0.5))
        lim = float(np.nanmax(np.abs(samples[error_name])))
        kwargs.setdefault('scalar_bar_args', {'title': f'{error_name} (hold-out)'})
        return self._plotter.add_mesh(
            glyphs, scalars=error_name, cmap=cmap, clim=(-lim, lim), **kwargs
        )

    def add_anisotropy_ellipsoid(
        self,
        model: KrigingModel,
        *,
        center: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ) -> pv.Actor:
        """Add the ellipsoid of the variogram range, stretched by the model's anisotropy.

        Samples inside the ellipsoid around a point are correlated with it.
        Its semi-axes are the range along the major direction and the range
        over ``anisotropy_scaling_y`` and ``anisotropy_scaling_z`` along the
        others, rotated by the anisotropy angles as PyKrige applies them. A 2D
        model gets a flat ellipse.

        Parameters
        ----------
        model : KrigingModel
            Model with a bounded variogram (spherical, exponential, gaussian or
            hole-effect).

        center : sequence[float], optional
            Center of the ellipsoid. Defaults to the centroid of the samples.

        **kwargs : dict, optional
            Options forwarded to :meth:`pyvista.Plotter.add_mesh`. The default
            style is a translucent surface.

        Returns
        -------
        pyvista.Actor
            The actor of the ellipsoid.

        """
        import numpy as np  # noqa: PLC0415

        vario = model.variogram
        if vario.model_name not in _BOUNDED:
            msg = (
                f'The {vario.model_name!r} variogram has no range. Use a spherical, '
                'exponential, gaussian or hole-effect model.'
            )
            raise ValueError(msg)
        rng = float(vario.parameters[1])
        instance = model.instance
        if model.dim == 3:
            scaling = (1.0, instance.anisotropy_scaling_y, instance.anisotropy_scaling_z)
            angles = np.radians(
                [
                    instance.anisotropy_angle_x,
                    instance.anisotropy_angle_y,
                    instance.anisotropy_angle_z,
                ]
            )
        else:
            scaling = (1.0, instance.anisotropy_scaling, 1e3)
            angles = np.radians([0.0, 0.0, instance.anisotropy_angle])
        rotation = _pykrige_rotation(angles)
        ellipsoid = pv.ParametricEllipsoid(*(rng / np.asarray(scaling, dtype=float)))
        if center is None:
            center = model.samples.points.mean(axis=0)
        # PyKrige maps a point x to S @ R @ (x - c); invert the rotation for the axes.
        ellipsoid.points = ellipsoid.points @ rotation + np.asarray(center, dtype=float)
        kwargs.setdefault('opacity', 0.3)
        kwargs.setdefault('color', 'gold')
        return self._plotter.add_mesh(ellipsoid, **kwargs)


def _pykrige_rotation(angles: Any) -> Any:
    """Return PyKrige's anisotropy rotation ``Rz @ Ry @ Rx`` for angles in radians."""
    import numpy as np  # noqa: PLC0415

    ax, ay, az = -np.asarray(angles)
    rotate_x = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rotate_y = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rotate_z = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rotate_z @ rotate_y @ rotate_x
