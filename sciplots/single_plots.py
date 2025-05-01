"""
Plots that use a single axis
"""
import logging
from typing import Any

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from sciplots.base import BasePlot


class BaseSinglePlot(BasePlot):
    """
    Base class to plot multiple sets of data on the same axes

    Attributes
    ----------
    plots : dict[Axes, list[Artist | Container]], default = {}
        Plot artists for each axis
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
    _alpha_line: float = 1
    _marker_size: float = 200

    def __init__(
            self,
            x_data: list[ndarray] | ndarray,
            y_data: list[ndarray] | ndarray,
            log_x: bool = False,
            log_y: bool = False,
            error_region: bool = False,
            x_label: str = '',
            y_label: str = '',
            markers: str | list[str] = 'x',
            labels: list[str] | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        x_data : list[ndarray] | ndarray
            List of B sets of x-values with ndarray shape N; or ndarray with shape (B,N)
        y_data : list[ndarray] | ndarray
            List of B sets of y-values with ndarray shape N; or ndarray with shape (B,N)
        log_x : bool, default = False
            If the x-axis should be logarithmic
        log_y : bool, default = False
            If the y-axis should be logarithmic
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
        x_error : list[ndarray] | ndarray | None, default = None
            List of B sets of x-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        y_error : list[ndarray] | ndarray | None, default = None
            List of B sets of y-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log_x: bool = log_x
        self._log_y: bool = log_y
        self._marker: list[str]
        self._data: list[ndarray] | ndarray
        self._y_data: list[ndarray] | ndarray
        self._x_error: list[ndarray] | list[None] | ndarray
        self._y_error: list[ndarray] | list[None] | ndarray
        self._error_region = error_region

        (self._data,
         (self._markers, self._labels),
         (self._y_data, self._x_error, self._y_error)) = self._data_length_normalise(
            x_data,
            lists=[markers, labels],
            data=[y_data, x_error, y_error],
        )

        if len(self._x_error) == 2 and len(self._data) != 2:
            self._x_error = [self._x_error] * len(self._data)

        if len(self._y_error) == 2 and len(self._data) != 2:
            self._y_error = [self._y_error] * len(self._data)

        if self._error_region and self._x_error[0]:
            logging.getLogger(__name__).warning('X-error is ignored if error_region is True')

        super().__init__(
            self._data,
            x_label=x_label,
            y_label=y_label,
            labels=self._labels,
            **kwargs,
        )

    def _axis_plot_data(
            self,
            labels: list[str],
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
        x_data : list[ndarray] | ndarray
            List of B sets of x-values with ndarray shape N; or ndarray with shape (B,N)
        y_data : list[ndarray] | ndarray
            List of B sets of y-values with ndarray shape N; or ndarray with shape (B,N)
        x_error : list[ndarray] | list[None] | ndarray
            List of B sets of x-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        y_error : list[ndarray] | list[None] | ndarray
            List of B sets of y-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        axis : Axes
            Axis to plot the data on

        **kwargs
            Optional keyword arguments to pass to self.plot_errors
        """
        label: str
        colour: str
        marker: str
        x_datum: ndarray
        y_datum: ndarray
        x_error: ndarray
        y_error: ndarray

        for label, colour, marker, x_datum, y_datum, x_error, y_error \
                in zip(labels, self._colours, self._markers, x_data, y_data, x_errors, y_errors):
            self.plot_errors(
                colour,
                x_datum,
                y_datum,
                axis,
                label=label,
                marker=marker,
                x_error=x_error,
                y_error=y_error,
                **kwargs,
            )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self.axes.set_xscale('log' if self._log_x else 'linear')
        self.axes.set_yscale('log' if self._log_y else 'linear')
        self._axis_plot_data(
            self._labels,
            self._data,
            self._y_data,
            self._x_error,
            self._y_error,
            self.axes,
        )

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
        y_data : list[ndarray] | ndarray
            List of B sets of y-values with ndarray shape N; or ndarray with shape (B,N)
        log_y : bool, default = False
            If the y-axis should be logarithmic
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of data
        x_data : list[ndarray] | ndarray | None, default = None
            List of B sets of x-values with ndarray shape N; or ndarray with shape (B,N), if None,
            original x-values will be used
        x_error : list[ndarray] | ndarray | None, default = None
            List of B sets of x-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        y_error : list[ndarray] | ndarray | None, default = None
            List of B sets of y-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        """
        assert isinstance(self.axes, Axes)
        axis: Axes = self.axes.twinx()
        x_data = x_data or self._data
        self._axis_init(axis)
        axis.set_yscale('log' if log_y else 'linear')
        axis.set_ylabel(y_label, fontsize=self._major)
        x_data, _, (y_data, x_error, y_error, labels) = self._data_length_normalise(
            x_data,
            data=[y_data, x_error, y_error, labels],
        )

        self._axis_plot_data(labels, x_data, y_data, x_error, y_error, axis, linestyle='--')
        self.set_axes_pad()

        if labels[0] is not None:
            self._labels = self._labels + labels if self._labels is not None else labels
            self.create_legend(**self._legend_kwargs)


class PlotComparison(BaseSinglePlot):
    """
    Plots a comparison between x & y data

    Attributes
    ----------
    plots : dict[Axes, list[Artist | Container]], default = {}
        Plot artists for each axis
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
            error_region: bool = False,
            x_label: str = '',
            y_label: str = '',
            residual: str | None = None,
            labels: list[str] | None = None,
            target: list[ndarray] | ndarray | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        x_data : list[ndarray] | ndarray
            List of B sets of x-values with ndarray shape N; or ndarray with shape (B,N)
        y_data : list[ndarray] | ndarray
            List of B sets of y-values with ndarray shape N; or ndarray with shape (B,N)
        log_x : bool, default = False
            If the x-axis should be logarithmic
        log_y : bool, default = False
            If the y-axis should be logarithmic
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
        x_error : list[ndarray] | ndarray | None, default = None
            List of B sets of x-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors
        y_error : list[ndarray] | ndarray | None, default = None
            List of B sets of y-errors with ndarray shape N or shape (2,N) for asymmetric errors; or
            ndarray with shape (N) if all input data has the same error, (2,N) for asymmetric
            errors, (B,N) for unique errors, or (B,2,N) for unique asymmetric errors

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
        *_, (self._target,) = self._data_length_normalise(self._data, data=[self._target])

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        label: str
        colour: str
        marker: str
        pred: ndarray
        x_data: ndarray
        target: ndarray
        x_error: ndarray
        y_error: ndarray
        axis: Axes

        self.axes.set_xscale('log' if self._log_x else 'linear')

        for label, colour, marker, x_data, pred, target, x_error, y_error in zip(
                self._labels,
                self._colours,
                self._markers,
                self._data,
                self._y_data,
                self._target,
                self._x_error,
                self._y_error):
            if self._residual is None:
                self.plot_errors(
                    colour,
                    x_data,
                    pred,
                    self.axes,
                    label=label,
                    marker=marker,
                    x_error=x_error,
                    y_error=y_error,
                )
                self.plots[self.axes].append(self.axes.plot(x_data, target, color='k')[0])
            else:
                self._major_axes = self.plot_residuals(
                    colour,
                    x_data,
                    pred,
                    target,
                    self.axes,
                    label=label,
                    error=self._residual.lower() == 'error',
                    target_colour='k',
                    x_error=x_error,
                    y_error=y_error,
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

        if len(self._data) == 1 and self._y_error[0] is not None and np.ndim(self._y_error[0]) == 1:
            axis = self._major_axes or self.axes
            self.plots[axis].append(axis.text(
                0.1,
                0.9,
                r'$\chi^2_\nu=$'
                f'{np.mean((self._y_data[0] - self._target[0]) ** 2 / self._y_error[0] ** 2):.2f}',
                fontsize=self._minor,
                transform=(self._major_axes or self.axes).transAxes,
            ))


class PlotPerformance(BaseSinglePlot):
    """
    Plots the performance over epochs of training

    Attributes
    ----------
    plots : dict[Axes, list[Artist | Container]], default = {}
        Plot artists for each axis
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
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            x_data: list[float] | list[ndarray] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[ndarray] | ndarray
            List of B sets of performance metrics with ndarray shape N; or ndarray with shape (B,N)
        log : bool, default = True
            If the y-axis should be logarithmic
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of performance metrics
        x_data : list[ndarray] | ndarray | None, default = None
            List of B sets of x-values with ndarray shape N; or ndarray with shape (B,N)

        **kwargs
            Optional keyword arguments to pass to PlotPlots
        """
        self._y_data: list[ndarray] | ndarray = [data] if np.ndim(data[0]) < 1 else data
        super().__init__(
            x_data or [np.arange(len(datum)) for datum in self._y_data],
            self._y_data,
            log_y=log,
            x_label=x_label,
            y_label=y_label,
            markers='',
            labels=labels,
            **kwargs,
        )

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
                fontsize=self._minor,
                transform=self.axes.transAxes
            )


class PlotPlots(BaseSinglePlot):
    """
    Plots multiple datasets on the same axes

    Attributes
    ----------
    plots : dict[Axes, list[Artist | Container]], default = {}
        Plot artists for each axis
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend
    """
