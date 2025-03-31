"""
Plots that compare each parameter within two sets of data between each other
"""
from typing import Any

import numpy as np
from numpy import ndarray

from sciplots import utils
from sciplots.base import BasePlot


class BaseParamPairs(BasePlot):
    """
    A class for creating parameter pair plots

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
    _alpha_marker: float = 0.2

    def __init__(
            self,
            x_data: list[ndarray] | ndarray,
            density: bool = False,
            labels: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            y_data: list[ndarray] | ndarray | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        x_data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters
        density : bool, default = False
            If the data should be plotted as contours and density sciplots or histograms
        labels : list[str] | None, default = None
            Label for each set of parameter comparisons
        x_labels : list[str] | None, default = None
            Labels for each x-axis parameter
        y_labels : list[str] | None, default = None
            Labels for each y-axis parameter
        y_data : list[(N,P) ndarray] | (N,P) ndarray | (B,N,P) ndarray | None, default = None
            B sets of N data points for P parameters to compare against x-axis parameters
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._labels: list[str] = labels or ['']
        self._x_labels: list[str] | None = x_labels
        self._y_labels: list[str] | None = y_labels
        self._data: list[ndarray]
        self._y_data: list[ndarray]
        self._subplot_kwargs: dict[str, Any] = {}
        self._x_ranges: ndarray
        self._y_ranges: ndarray

        self._data, self._x_ranges = self._preprocessing(x_data)
        self._y_data, self._y_ranges = self._preprocessing(x_data if y_data is None else y_data)

        if len(self._data) == 1:
            self._data *= len(self._y_data)

        if len(self._y_data) == 1:
            self._y_data *= len(self._data)

        if len(self._labels) == 1:
            self._labels *= len(self._data)

        super().__init__(
            self._data,
            density=density,
            labels=self._labels,
            fig_size=fig_size,
            **kwargs,
        )

    @staticmethod
    def _preprocessing(data: list[ndarray] | ndarray) -> tuple[list[ndarray], ndarray]:
        """
        Preprocess the input data for plotting

        Parameters
        ----------
        data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters

        Returns
        -------
        tuple[list[(L,N) ndarray], (L,2) ndarray]
            B sets of N preprocessed data points for L parameters and minimum and maximum values for
            the L parameters
        """
        pad: float = 0.05
        ranges: ndarray

        data = [data] if np.ndim(data[0]) < 2 else data
        ranges = np.stack((
            np.min([np.min(datum - np.abs(pad * datum), axis=0) for datum in data], axis=0),
            np.max([np.max(datum + np.abs(pad * datum), axis=0) for datum in data], axis=0),
        ), axis=1)
        return list(data), ranges

    def _axes_init(self) -> None:
        self.subplots(
            (self._y_data[0].shape[-1], self._data[0].shape[-1]),
            borders=True,
            x_labels=self._x_labels,
            y_labels=self._y_labels,
            **self._subplot_kwargs,
        )
        self.axes[0, 0].set_ylabel('')


class PlotParamPairs(BaseParamPairs):
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

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
            data: list[ndarray] | ndarray,
            density: bool = False,
            labels: list[str] | None = None,
            axes_labels: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters
        density : bool, default = False
            If the data should be plotted as contours and density sciplots or histograms
        labels : list[str] | None, default = None
            Labels for each set of parameter comparisons
        axes_labels : list[str] | None, default = None
            Labels for the axes
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._subplot_kwargs = {'sharex': 'col'}
        super().__init__(
            data,
            density=density,
            labels=labels,
            x_labels=axes_labels,
            y_labels=axes_labels,
            fig_size=fig_size,
            **kwargs,
        )

    def _plot_data(self) -> None:
        label: str
        colour: str
        data: ndarray

        for label, colour, data in zip(self._labels, self._colours, self._data):
            self.plot_param_pairs(
                colour,
                data,
                label=label,
                ranges=self._x_ranges,
            )


class PlotParamPairComparison(BaseParamPairs):
    """
    Plots the comparison between y-data and x-data

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
            density: bool = False,
            labels: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        x_data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters
        y_data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters to compare against x_data
        density : bool, default = False
            If the data should be plotted as contours and density sciplots or histograms
        labels : list[str] | None, default = None
            Labels for each set of parameter comparisons
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        x_labels = x_labels + [''] if x_labels else None
        y_labels = [''] + y_labels if y_labels else None
        super().__init__(
            x_data,
            density=density,
            labels=labels,
            x_labels=x_labels,
            y_labels=y_labels,
            y_data=y_data,
            fig_size=fig_size,
            **kwargs,
        )

    def _axes_init(self) -> None:
        self.subplots(
            (self._y_data[0].shape[-1] + 1, self._data[0].shape[-1] + 1),
            borders=True,
            x_labels=self._x_labels,
            y_labels=self._y_labels,
        )

    def _plot_data(self) -> None:
        label: str
        colour: str
        x_data: list[ndarray]
        y_data: list[ndarray]
        datum: ndarray
        x_datum: ndarray
        y_datum: ndarray

        assert isinstance(self.axes, ndarray)
        self.axes[0, -1].set_visible(False)

        x_data = [np.swapaxes(datum, 0, 1) for datum in self._data]
        y_data = [np.swapaxes(datum, 0, 1) for datum in self._y_data]

        for label, colour, x_datum, y_datum in zip(
                self._labels,
                self._colours,
                x_data,
                y_data):
            for i, (axes_row, row_data, row_range) in enumerate(zip(
                    self.axes[1:],
                    y_datum,
                    self._y_ranges)):
                for j, (axis, col_data, col_range) in enumerate(zip(
                        axes_row,
                        x_datum,
                        self._x_ranges)):
                    axis.set_xlim(col_range)
                    axis.set_ylim(row_range)

                    # Set x & y labels and hide ticks for sciplots that aren't in the first column
                    # or bottom row
                    if j != 0:
                        axis.tick_params(labelleft=False, left=False)

                    if i != len(self.axes) - 2:
                        axis.tick_params(labelbottom=False, bottom=False)

                    # Set number of ticks
                    axis.locator_params(axis='x', nbins=3)
                    axis.locator_params(axis='y', nbins=3)

                    # Plot histograms
                    if i == 0:
                        self.axes[0, j].set_xlim(col_range)
                        self.plot_hist(colour, col_data, self.axes[0, j], range_=col_range)
                        self.axes[0, j].tick_params(
                            labelbottom=False,
                            bottom=False,
                            labelleft=False,
                            left=False,
                        )

                    if j == self.axes.shape[1] - 2:
                        axes_row[-1].set_ylim(row_range)
                        self.plot_hist(
                            colour,
                            row_data,
                            axes_row[-1],
                            label=label,
                            orientation='horizontal',
                            range_=row_range,
                        )
                        axes_row[-1].tick_params(
                            labelbottom=False,
                            bottom=False,
                            labelleft=False,
                            left=False,
                        )

                    # Plot scatter sciplots
                    self.plots[axis].append(axis.scatter(
                        col_data[:self._scatter_num],
                        row_data[:self._scatter_num],
                        s=4,
                        alpha=self._alpha_marker,
                        color=colour,
                        label=label,
                    ))

                    if (self._density and
                            len(np.unique(col_data)) > 1 and len(np.unique(row_data)) > 1):
                        self.plot_density(
                            colour,
                            np.stack((col_data, row_data), axis=1),
                            np.array((col_range, row_range)),
                            axis=axis,
                            label=label,
                        )
