"""
Distribution plotting classes
"""
from typing import Any

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from sciplots.base import BasePlot
from sciplots.utils import subplot_grid


class BaseDistribution(BasePlot):
    """
    Base class to plot distributions of data

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
    plot_vlines(x_data, label='', colour='') -> None
        Plots vertical lines on each distribution
    """
    def __init__(
            self,
            data: list[ndarray] | ndarray,
            norm: bool = False,
            single: bool = True,
            y_axes: bool = False,
            density: bool = False,
            log: bool | list[bool] = False,
            bins: int = 100,
            labels: str | list[str] | ndarray = '',
            hatches: str | list[str] | ndarray = '',
            titles: list[str] | ndarray | None = None,
            x_labels: list[str] | ndarray | None = None,
            y_labels: list[str] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        norm : bool, default = False
            If the distributions should be normalised to a maximum height of 1
        single : bool, default = True
            If the distributions should be plotted on a single plot
        y_axes : bool, default = False
            If the y-axis should have ticks
        density : bool, default = False
            If the distributions should be histograms or density plots
        log : bool | list[bool], default = False
            If the x-axis should be logarithmic
        bins : int, default = 100
            Number of bins for histograms or density interpolation
        labels : str | list[str] | ndarray, default = ''
            Labels for the data
        hatches : str | list[str] | ndarray, default = ''
            Hatches for the data
        titles : list[str] | ndarray | None, default = None
            Titles for the distributions
        x_labels : list[str] | ndarray | None, default = None
            X-axis label(s)
        y_labels : list[str] | ndarray | None, default = None
            Y-axis label(s)

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._norm: bool
        self._log: list[bool]
        self._single: bool = single
        self._y_axes: bool = y_axes
        self._hatches: list[str]
        self._ranges: list[tuple[float, float]]
        self._data: list[ndarray] | ndarray = [data] if np.ndim(data[0]) < 1 else data

        self._data, (
            self._log,
            self._labels,
            self._hatches,
            self._titles,
            self._x_labels,
            self._y_labels,
        ), _ = self._data_length_normalise(self._data, lists=[
            log,
            labels.tolist() if isinstance(labels, ndarray) else labels,
            hatches.tolist() if isinstance(hatches, ndarray) else hatches,
            titles.tolist() if isinstance(titles, ndarray) else titles,
            x_labels.tolist() if isinstance(x_labels, ndarray) else x_labels,
            y_labels.tolist() if isinstance(y_labels, ndarray) else y_labels,
        ])
        self._norm = True if any(datum.size == 1 for datum in self._data) else norm

        super().__init__(
            self._data,
            density=density,
            bins=bins,
            x_label=self._x_labels[0] if self._single else '',
            y_label=self._y_labels[0] if self._single else '',
            labels=self._labels,
            **kwargs,
        )

    def _axes_init(self) -> None:
        if self._single:
            super()._axes_init()
            return

        self.subplots(
            subplot_grid(len(self._data)),
            titles=self._titles,
            x_labels=self._x_labels,
            y_labels=self._y_labels,
        )

    def _axis_plot_data(
            self,
            log: bool,
            labels: list[str],
            colours: list[str],
            hatches: list[str],
            data: list[ndarray] | ndarray,
            axis: Axes) -> None:
        """
        Plots the data for a given axis

        Parameters
        ----------
        log : bool
            If the x-axis should be logarithmic
        labels : list[str]
            Labels for the data
        colours : list[str]
            Colours for the data
        hatches : list[str]
            Hatches for the data
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        axis : Axes
            Axis to plot the data
        """
        label: str
        hatch: str
        colour: str
        range_: tuple[float, float] | None = None if data[0].dtype.type == np.str_ else (
            float(np.min([np.min(datum) for datum in data])),
            float(np.max([np.max(datum) for datum in data])),
        )
        datum: ndarray
        axis.set_xscale('log' if log else 'linear')

        for label, hatch, colour, datum in zip(labels, hatches, colours, data):
            self.plot_hist(
                colour,
                datum,
                axis,
                log=log,
                norm=self._norm,
                label=label,
                hatch=hatch,
                range_=range_,
            )

        if not self._y_axes:
            axis.tick_params(labelleft=False, left=False)

    def plot_vlines(self, x_data: list[float] | ndarray, label: str = '', colour: str = '') -> None:
        """
        Plots vertical lines on each distribution

        Parameters
        ----------
        x_data : list[float] | (P) ndarray
            X-values for the vertical lines for the P plots
        label : str, default = ''
            Label for the vertical lines
        colour : str, default = ''
            Colour for the vertical lines, if empty, self._colours will be used
        """
        for x_datum, axis in zip(
                x_data,
                [self.axes] if isinstance(self.axes, Axes) else
                self.axes.values() if isinstance(self.axes, dict) else
                self.axes.flatten()):
            ranges = self._axis_data_ranges(axis)

            if ranges is None:
                continue

            self.plots[axis].append(axis.vlines(
                x_datum,
                *ranges[:, 1],
                linewidths=2,
                label=label,
                colors=colour if colour else 'k',
            ))

        self.set_axes_pad()

        if label:
            self._labels = self._labels + [label] if self._labels is not None else [label]
            self.create_legend(**self._legend_kwargs)


class PlotDistribution(BaseDistribution):
    """
    PLots multiple distributions on a single plot

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
    plot_twin_data(data, y_label='', labels=None, colours=None, hatches=None) -> None
        Plots distributions with the same x-axis on a different y-axis
    """
    def __init__(
            self,
            data: list[ndarray] | ndarray,
            log: bool = False,
            norm: bool = False,
            y_axes: bool = False,
            density: bool = False,
            bins: int = 100,
            x_labels: str = '',
            y_labels: str = '',
            labels: str | list[str] | ndarray = '',
            hatches: str | list[str] | ndarray = '',
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        log : bool, default = False
            If the x-axis should be logarithmic
        norm : bool, default = False
            If the distributions should be normalised to a maximum height of 1
        y_axes : bool, default = False
            If the y-axis should have ticks
        density : bool, default = False
            If the distributions should be histograms or density plots
        bins : int, default = 100
            Number of bins for histograms or density interpolation
        x_labels : str, default = ''
            X-axis label
        y_labels : str, default = ''
            Y-axis label
        labels : str | list[str] | ndarray, default = ''
            Labels for the data
        hatches : str | list[str] | ndarray, default = ''
            Hatches for the data

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        super().__init__(
            data,
            norm=norm,
            single=True,
            y_axes=y_axes,
            density=density,
            log=log,
            bins=bins,
            labels=labels,
            hatches=hatches,
            titles=None,
            x_labels=[x_labels],
            y_labels=[y_labels],
            **kwargs,
        )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self._axis_plot_data(
            self._log[0],
            self._labels,
            self._colours,
            self._hatches,
            self._data,
            self.axes,
        )

    def plot_twin_data(
            self,
            data: list[ndarray] | ndarray,
            y_label: str = '',
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            hatches: list[str] | None = None) -> None:
        """
        Plots distributions with the same x-axis on a different y-axis

        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        y_label : str, default = ''
            Y-axis label for the twin axis
        labels : list[str] | None, default = None
            Labels for the data
        colours : list[str] | None, default = None
            Colours for the data, if None, default colours will be used
        hatches : list[str] | None, default = None
            Hatches for the data
        """
        axis: Axes = self._twin_axes(labels=y_label)
        colours = colours or self._colours
        data = [data] if np.ndim(data[0]) < 1 else data

        data, (labels, hatches), _ = self._data_length_normalise(data, lists=[labels, hatches])
        self._axis_plot_data(self._log[0], self._labels, colours, hatches, data, axis)
        self.set_axes_pad()

        if labels:
            self._labels += labels
            self.create_legend(**self._legend_kwargs)


class PlotDistributions(BaseDistribution):
    """
    Plots distributions on different subplots

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
    plot_twin_data(data, y_label='', labels=None, colours=None, hatches=None) -> None
        Plots distributions with the same x-axis on a different y-axis
    """
    def __init__(
            self,
            data: list[ndarray] | ndarray,
            norm: bool = False,
            y_axes: bool = True,
            density: bool = False,
            log: bool | list[bool] = False,
            bins: int = 100,
            label: str = '',
            hatches: str | list[str] | ndarray = '',
            x_labels: str | list[str] | ndarray = '',
            y_labels: str | list[str] | ndarray = '',
            titles: list[str] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        norm : bool, default = False
            If the distributions should be normalised to a maximum height of 1
        y_axes : bool, default = False
            If the y-axis should have ticks
        density : bool, default = False
            If the distributions should be histograms or density plots
        log : bool | list[bool], default = False
            If the x-axis should be logarithmic
        bins : int, default = 100
            Number of bins for histograms or density interpolation
        label : str, default = ''
            Label for the data
        hatches : str | list[str] | ndarray, default = ''
            Hatches for the data
        x_labels : str | list[str] | ndarray, default = ''
            X-axis label(s)
        y_labels : str | list[str] | ndarray, default = ''
            Y-axis label(s)
        titles : list[str] | ndarray | None, default = None
            Titles for the distributions

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        super().__init__(
            data,
            norm=norm,
            single=False,
            y_axes=y_axes,
            density=density,
            log=log,
            bins=bins,
            labels=label,
            hatches=hatches,
            x_labels=x_labels,
            y_labels=y_labels,
            titles=titles,
            **kwargs,
        )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        log: bool
        datum: ndarray
        axis: Axes

        for log, datum, axis in zip(self._log, self._data, self.axes.values()):
            self._axis_plot_data(
                log,
                self._labels[:1],
                self._colours[:1],
                self._hatches[:1],
                [datum],
                axis,
            )

    def plot_twin_data(
            self,
            data: list[ndarray] | ndarray,
            label: str = '',
            hatch: str = '',
            colour: str = '',
            y_labels: str | list[str] = '') -> None:
        """
        Plots distributions with the same x-axis on a different y-axis

        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        label : str, default = ''
            Label for the twin data
        hatch : str, default = ''
            Hatch for the twin data
        colour : str, default = ''
            Colour for the twin data, if None, default colours will be used
        y_labels : str | list[str], default = ''
            Y-axis label(s)

        Returns
        -------

        """
        log: bool
        axes: dict[int | str, Axes] | ndarray
        datum: ndarray
        axis: Axes
        data = [data] if np.ndim(data[0]) < 1 else data

        data, (y_labels,), _ = self._data_length_normalise(data, lists=[y_labels])
        axes = self._twin_axes(labels=y_labels)

        for log, datum, axis in zip(self._log, data, axes.values()):
            self._axis_plot_data(
                log,
                [label],
                [colour] if colour else self._colours[1:2],
                [hatch] or (self._hatches[1:2] if len(self._hatches) > 1 else self._hatches[:1]),
                [datum],
                axis,
            )

        if label:
            self._labels += [label]
            self.create_legend(**self._legend_kwargs)
