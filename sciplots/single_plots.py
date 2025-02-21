"""
Plots that use a single axis
"""
import logging
from typing import Any, Callable

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from sciplots import utils
from sciplots.base import BasePlot


class _BaseSinglePlot(BasePlot):
    """
    Base class to plot multiple sets of data on the same axes

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend

    Methods
    -------
    plot_twin_data(y_data, log_y, y_label, labels, x_data, x_error, y_error)
        Plots data with the same x-axis on a different y-axis
    """
    def __init__(
            self,
            x_data: list[ndarray] | ndarray,
            y_data: list[ndarray] | ndarray,
            log_x: bool = False,
            log_y: bool = False,
            axis_pad: bool = True,
            error_region: bool = False,
            x_label: str = '',
            y_label: str = '',
            markers: str | list[str] = 'x',
            labels: list[str] | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        x_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N x-values to plot
        y_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N y-values to plot
        log_x : bool, default = False
            If the x-axis should be logarithmic
        log_y : bool, default = False
            If the y-axis should be logarithmic
        axis_pad : bool, default = True
            If the axis should have padding between the axis and the data points
        error_region : bool, default = False
            If the errors should be error bars or a highlighted region, highlighted region only
            supports y_errors
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        markers : str | list[str], default = 'x'
            Marker style for the data or each set of data, if '', plot will be used
        labels : list[str] | None, default = None
            Labels for each set of data
        x_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-errors to plot
        y_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N y-errors to plot

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log_x: bool = log_x
        self._log_y: bool = log_y
        self._axis_pad: bool = axis_pad
        self._error_region: bool = error_region
        self._marker: list[str]
        self._data: list[ndarray] | ndarray = [x_data] if np.ndim(x_data[0]) < 1 else x_data
        self._y_data: list[ndarray] | ndarray = [y_data] if np.ndim(y_data[0]) < 1 else y_data
        self._x_error: list[ndarray] | list[None] | ndarray
        self._y_error: list[ndarray] | list[None] | ndarray

        if len(self._data) == 1:
            self._data = [self._data[0]] * len(self._y_data)

        if len(self._y_data) == 1:
            self._y_data = [self._y_data[0]] * len(self._data)

        if isinstance(markers, str):
            self._markers = [markers] * len(self._data)
        else:
            self._markers = markers

        if x_error is None:
            self._x_error = [None] * len(self._data)
        else:
            self._x_error = [x_error] if np.ndim(x_error[0]) < 1 else x_error

        if y_error is None:
            self._y_error = [None] * len(self._y_data)
        else:
            self._y_error = [y_error] if np.ndim(y_error[0]) < 1 else y_error

        if self._error_region and self._x_error[0]:
            logging.getLogger(__name__).warning('X-error is ignored if error_region is True')

        super().__init__(
            self._data,
            x_label=x_label,
            y_label=y_label,
            labels=labels,
            **{'alpha': 1} | kwargs,
        )

    @staticmethod
    def _remove_axis_pad(
            log: bool,
            x_axis: bool,
            data: list[ndarray] | ndarray,
            errors: list[ndarray | None] | ndarray,
            axis: Axes) -> None:
        """
        Removes axis padding from the x-axis or y-axis

        Parameters
        ----------
        log : bool
            If axis is logged
        x_axis : bool
            Whether to use the x or y-axis
        data : list[ndarray] | ndarray
            Data to set the limits for
        errors : list[ndarray | None] | ndarray
            Errors to add and subtract from the data
        axis : Axes
            Axis to set the limits for
        """
        set_lim: Callable = axis.set_xlim if x_axis else axis.set_ylim
        mins: list[float] = []
        maxs: list[float] = []
        where: ndarray
        datum: ndarray
        error: ndarray

        for datum, error in zip(data, errors):
            error = 0 if error is None else error
            where = datum - error > 0 if log else np.array([True])
            maxs.append(np.max(datum + error))
            mins.append(np.min(datum - error, initial=maxs[-1], where=where))

        set_lim([min(mins), max(maxs)])

    def _post_init(self) -> None:
        self._default_name = 'sciplots'

        if self._axis_pad:
            return

        if self._data[0].dtype != str:
            self._remove_axis_pad(self._log_x, True, self._data, self._x_error, self.axes)

        if self._y_data[0].dtype != str:
            self._remove_axis_pad(self._log_y, False, self._y_data, self._y_error, self.axes)

    def _axis_plot_data(
            self,
            x_data: list[ndarray] | ndarray,
            y_data: list[ndarray] | list[None] | ndarray,
            x_errors: list[ndarray] | list[None] | ndarray,
            y_errors: list[ndarray] | list[None] | ndarray,
            axis: Axes,
            **kwargs: Any) -> None:
        """
        Plots the data for a given axis

        Parameters
        ----------
        x_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N x-values to plot
        y_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N y-values to plot
        x_errors : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-errors to plot
        y_errors : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N y-errors to plot
        axis : Axes
            Axis to plot the data on

        **kwargs
            Optional keyword arguments to pass to self.plot_errors
        """
        colour: str
        marker: str
        x_datum: ndarray
        y_datum: ndarray
        x_error: ndarray
        y_error: ndarray

        for colour, marker, x_datum, y_datum, x_error, y_error \
                in zip(self._colours, self._markers, x_data, y_data, x_errors, y_errors):
            self.plot_errors(
                colour,
                x_datum,
                y_datum,
                axis,
                marker=marker,
                error_region=self._error_region,
                x_error=x_error,
                y_error=y_error,
                **kwargs,
            )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self.axes.set_xscale('log' if self._log_x else 'linear')
        self.axes.set_yscale('log' if self._log_y else 'linear')
        self._axis_plot_data(self._data, self._y_data, self._x_error, self._y_error, self.axes)

    def plot_twin_data(
            self,
            y_data: list[ndarray] | ndarray,
            log_y: bool = False,
            y_label: str = '',
            labels: list[str] | None = None,
            x_data: list[ndarray] | ndarray | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None) -> None:
        """
        Plots data with the same x-axis on a different y-axis

        Parameters
        ----------
        y_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N y-values to plot
        log_y : bool, default = False
            If the y-axis should be logarithmic
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of data
        x_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-values to plot, if None, original x-values will be used
        x_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-errors to plot
        y_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N y-errors to plot
        """
        assert isinstance(self.axes, Axes)
        axis: Axes = self.axes.twinx()
        x_data = x_data or self._data
        x_data = [x_data] if np.ndim(x_data[0]) < 1 else x_data
        y_data = [y_data] if np.ndim(y_data[0]) < 1 else y_data
        self._axis_init(axis)
        axis.set_yscale('log' if log_y else 'linear')
        axis.set_ylabel(y_label, fontsize=self._major)

        if len(x_data) == 1:
            x_data = [x_data[0]] * len(y_data)

        if len(y_data) == 1:
            y_data *= [y_data[0]] * len(self._data)

        if x_error is None:
            x_error = [None] * len(self._data)
        else:
            x_error = [x_error] if np.ndim(x_error[0]) < 1 else x_error

        if y_error is None:
            y_error = [None] * len(y_data)
        else:
            y_error = [y_error] if np.ndim(y_error[0]) < 1 else y_error

        self._axis_plot_data(x_data, y_data, x_error, y_error, axis, linestyle='--')

        if not self._axis_pad:
            if self._y_data[0].dtype != str:
                self._remove_axis_pad(log_y, False, y_data, y_error, axis)

        if labels is not None:
            self._labels = self._labels + labels if self._labels is not None else labels
            self.legend.remove()
            self.create_legend(**self._legend_kwargs)


class PlotComparison(_BaseSinglePlot):
    """
    Plots a comparison between x & y data

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend
    """
    def __init__(
            self,
            x_data: list[ndarray] | ndarray,
            y_data: list[ndarray] | ndarray,
            log_x: bool = False,
            log_y: bool = False,
            axis_pad: bool = True,
            error_region: bool = False,
            x_label: str = '',
            y_label: str = '',
            residual: str | None = None,
            labels: list[str] | None = None,
            target: list[ndarray] | ndarray | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        x_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N x-values to plot
        y_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N y-values to plot
        log_x : bool, default = False
            If the x-axis should be logarithmic
        log_y : bool, default = False
            If the y-axis should be logarithmic
        axis_pad : bool, default = True
            If the axis should have padding between the axis and the data points
        error_region : bool, default = False
            If the errors should be error bars or a highlighted region, highlighted region only
            supports y_errors
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        residual : {None, 'residual', 'error'}, default = None
            If the residuals should be plotted as residuals or errors
        labels : list[str] | None, default = None
            Labels for each set of data
        target : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N target values, if None, target is the x-values
        x_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-errors to plot
        y_error : [list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N y-errors to plot

        **kwargs
            Optional keyword arguments to pass to _BaseSinglePlot
        """
        self._residual: str | None = residual.lower() if isinstance(residual, str) else residual
        self._target: list[ndarray] | ndarray = x_data if target is None else target
        self._major_axes: Axes | None = None

        if self._residual not in {None, 'residual', 'error'}:
            raise ValueError(f'Invalid residual argument ({self._residual}), must be one of (None, '
                             f'residual, error)')

        super().__init__(
            x_data,
            y_data,
            log_x=log_x,
            log_y=log_y,
            axis_pad=axis_pad,
            error_region=error_region,
            x_label=x_label,
            y_label=y_label,
            labels=labels,
            x_error=x_error,
            y_error=y_error,
            **kwargs,
        )

    def _post_init(self) -> None:
        super()._post_init()
        self._default_name = 'comparison'
        self._target = [self._target] if np.ndim(self._target[0]) < 1 else self._target

        if len(self._target) == 1:
            self._target *= len(self._data)

        if self._axis_pad or self._y_data[0].dtype == str:
            return

        if self._residual:
            self._remove_axis_pad(
                self._log_y,
                False,
                [(data - target) / target if self._residual.lower() == 'error' else
                 data - target for data, target in zip(self._y_data, self._target)],
                [error / target if self._residual.lower() == 'error' else
                 error for error, target in zip(self._y_error, self._target)],
                self.axes,
            )
        else:
            self._remove_axis_pad(
                self._log_y,
                False,
                [*self._y_data, *self._target],
                [*self._y_error, *[None] * len(self._target)],
                self.axes,
            )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        colour: str
        marker: str
        pred: ndarray
        x_data: ndarray
        target: ndarray
        uncertainty: ndarray
        axis: Axes

        self.axes.set_xscale('log' if self._log_x else 'linear')

        for colour, marker, x_data, pred, target, uncertainty in zip(
                self._colours,
                self._markers,
                self._data,
                self._y_data,
                self._target,
                self._y_error):
            if self._residual is None:
                self.plot_errors(
                    colour,
                    x_data,
                    pred,
                    self.axes,
                    error_region=self._error_region,
                    marker=marker,
                    y_error=uncertainty,
                )
                self.axes.plot(x_data, target, color='k')
            else:
                self._major_axes = self.plot_residuals(
                    colour,
                    x_data,
                    pred,
                    target,
                    self.axes,
                    error=self._residual.lower() == 'error',
                    error_region=self._error_region,
                    target_colour='k',
                    uncertainty=uncertainty,
                    major_axis=self._major_axes,
                    marker=marker,
                )

        if self._major_axes:
            self._major_axes.set_xscale('log' if self._log_x else 'linear')
            self._major_axes.set_yscale('log' if self._log_y else 'linear')
            self._major_axes.set_xticks([])
            self._legend_axis = self._major_axes
        else:
            self.axes.set_yscale('log' if self._log_y else 'linear')

        if len(self._data) == 1 and self._y_error[0] is not None:
            axis = self._major_axes or self.axes
            axis.text(
                0.1,
                0.9,
                r'$\chi^2_\nu=$'
                f'{np.mean((self._y_data[0] - self._data[0]) ** 2 / self._y_error[0] ** 2):.2f}',
                fontsize=utils.MINOR,
                transform=(self._major_axes or self.axes).transAxes,
            )

        if self._axis_pad or self._major_axes is None:
            return

        if self._data[0].dtype != str:
            self._remove_axis_pad(
                self._log_x,
                True,
                self._data,
                self._x_error,
                self._major_axes,
            )

        if self._y_data[0].dtype != str:
            self._remove_axis_pad(
                self._log_y,
                False,
                [*self._y_data, *self._target],
                [*self._y_error, *[None] * len(self._target)],
                self._major_axes,
            )


class PlotDistribution(BasePlot):
    """
    Plots multiple sets of data onto a single distribution.

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend
    """
    def __init__(
            self,
            data: list[float] | list[ndarray] | ndarray,
            log: bool = False,
            norm: bool = False,
            y_axes: bool = True,
            density: bool = False,
            bins: int = 100,
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            hatches: list[str] | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        log : bool, default = False
            If the x-axis should be logarithmic
        norm : bool, default = False
            If the distributions should be normalised to a maximum height of 1
        y_axes : bool, default = True
            If the y-axis should have ticks
        density : bool, default = False
            If the distributions should be histograms or density sciplots
        bins : int, default = 100
            Number of bins for histograms or density interpolation
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of data
        hatches : list[str] | None, default = None
            Hatching pattern for each set of data

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log: bool = log
        self._norm: bool
        self._y_axes: bool = y_axes
        self._hatches: list[str]
        self._data: list[ndarray] | ndarray = [data] if np.ndim(data[0]) < 1 else data

        self._hatches = hatches or [''] * len(self._data)
        self._norm = True if any(datum.size == 1 for datum in self._data) else norm
        super().__init__(
            self._data,
            density=density,
            bins=bins,
            x_label=x_label,
            y_label=y_label,
            labels=labels,
            **kwargs,
        )

    def _post_init(self) -> None:
        self._default_name = 'distribution'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self.axes.set_xscale('log' if self._log else 'linear')
        range_: tuple[float, float] = (
            float(np.min([np.min(data) for data in self._data])),
            float(np.max([np.max(data) for data in self._data])),
        )

        for colour, hatch, data in zip(self._colours, self._hatches, self._data):
            self.plot_hist(
                colour,
                data,
                self.axes,
                log=self._log,
                norm=self._norm,
                hatch=hatch,
                range_=range_,
            )

        if not self._y_axes:
            self.axes.tick_params(labelleft=False, left=False)


class PlotPerformance(_BaseSinglePlot):
    """
    Plots the performance over epochs of training

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend
    """
    def __init__(
            self,
            data: list[float] | list[ndarray] | ndarray,
            log: bool = True,
            axis_pad: bool = True,
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            x_data: list[float] | list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[float] | list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N performance metrics over epochs
        log : bool, default = True
            If the y-axis should be logarithmic
        axis_pad : bool, default = True
            If the axis should have padding between the axis and the data points
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of performance metrics
        x_data : list[str] | list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-axis data points

        **kwargs
            Optional keyword arguments to pass to PlotPlots
        """
        self._y_data: list[ndarray] | ndarray = [data] if np.ndim(data[0]) < 1 else data
        super().__init__(
            x_data or [np.arange(len(datum)) for datum in self._y_data],
            self._y_data,
            log_y=log,
            axis_pad=axis_pad,
            x_label=x_label,
            y_label=y_label,
            markers='',
            labels=labels,
            **kwargs,
        )

    def _post_init(self) -> None:
        super()._post_init()
        self._default_name = 'performance'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        super()._plot_data()

        if self._labels and self._labels[0]:
            self.axes.text(
                0.7, 0.75,
                '\n'.join(
                    (f'Final {label}: {data[-1]:.3e}'
                     for label, data in zip(self._labels, self._y_data)),
                ),
                fontsize=utils.MINOR,
                transform=self.axes.transAxes
            )


class PlotPlots(_BaseSinglePlot):
    """
    Plots multiple datasets on the same axes

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend
    """
