"""
Miscellaneous plotting classes
"""
from typing import Any

import numpy as np
import matplotlib as mpl
from numpy import ndarray
from matplotlib.axes import Axes

from sciplots import utils
from sciplots.base import BasePlot
from sciplots.utils import subplot_grid


class PlotDistributions(BasePlot):
    """
    Plots the distributions of data points

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
    plot_twin_data(data, label, y_labels)
        Plots distributions with the same x-axis on a different y-axis
    """
    def __init__(
            self,
            data: list[ndarray] | ndarray,
            log: bool = False,
            norm: bool = False,
            y_axes: bool = True,
            density: bool = False,
            bins: int = 100,
            num_plots: int = 12,
            x_label: str = '',
            text_loc: str = 'upper right',
            labels: str | tuple[str, str] = '',
            hatches: str | tuple[str, str] = '',
            texts: list[str] | None = None,
            titles: list[str] | None = None,
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
        num_plots : int, default = 12
            Maximum number of distributions to plot
        x_label : str, default = ''
            X-axis label
        text_loc : str, default = 'upper right'
            Location of the text
        labels : str | tuple[str, str], default = ''
            Labels for the data and twin data
        hatches : str | tuple[str, str], default = ''
            Hatching pattern for the distributions and twin data
        texts : list[str] | None, default = None
            Texts to be displayed on the distributions
        titles : list[str] | None, default = None
            Titles for the distributions

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log: bool = log
        self._norm: bool
        self._y_axes: bool = y_axes
        self._num_plots: int
        self._text_loc: str = text_loc
        self._hatches: str | tuple[str, str] | None = hatches
        self._titles: list[str] | None = titles
        self._texts: list[str]
        self._data: list[ndarray] | ndarray =  [data] if np.ndim(data[0]) < 1 else data
        self._ranges: tuple[tuple[float, float], ...] = tuple(
            (np.min(datum), np.max(datum)) for datum in self._data
        )

        self._norm = True if any(datum.size == 1 for datum in self._data) else norm
        self._num_plots = min(num_plots, len(self._data))
        self._texts = texts or [''] * len(self._data)
        super().__init__(
            self._data,
            density=density,
            bins=bins,
            x_label=x_label,
            labels=list(labels) if isinstance(labels, tuple) else [labels],
            **kwargs,
        )

    def _axes_init(self) -> None:
        self.subplots(subplot_grid(self._num_plots), titles=self._titles)

    def _post_init(self) -> None:
        self._default_name = 'distributions'

    def _axis_plot_data(
            self,
            colour: str,
            hatch: str,
            data: list[ndarray] | ndarray,
            axes: list[Axes]) -> None:
        """
        Plots the data for a given axis

        Parameters
        ----------
        colour : str
            Colour of the distributions
        hatch : str
            Hatch pattern for the distributions
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N values to plot
        axes : list[Axes]
            Axis to plot the data on
        """
        range_: tuple[float, float]
        datum: ndarray
        axis: Axes

        for range_, datum, axis in zip(self._ranges, data, axes):
            axis.set_xscale('log' if self._log else 'linear')
            self.plot_hist(
                colour,
                datum,
                axis,
                log=self._log,
                norm=self._norm,
                hatch=hatch,
                range_=range_,
            )

            if not self._y_axes:
                axis.tick_params(labelleft=False, left=False)

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        text: str
        axis: Axes

        self._axis_plot_data(
            self._colours[0],
            self._hatches[0] if isinstance(self._hatches, tuple) else self._hatches,
            self._data,
            list(self.axes.values()),
        )

        for text, axis in zip(self._texts, self.axes.values()):
            if text:
                axis.add_artist(mpl.offsetbox.AnchoredText(
                    text,
                    loc=self._text_loc,
                    prop={'fontsize': utils.MINOR},
                ))

    def plot_twin_data(
            self,
            data: list[ndarray] | ndarray,
            label: str = '',
            y_labels: str | list[str] = '') -> None:
        """
        Plots distributions with the same x-axis on a different y-axis

        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N values to plot
        label : str, default = ''
            Label for the distributions
        y_labels : list[str] | None, default = ''
            Y-axis labels
        """
        y_label: str
        axes: list[Axes] = [axis.twinx() for axis in self.axes.values()]
        axis: Axes
        data = [data] if np.ndim(data[0]) < 1 else data
        y_labels = [y_labels] if isinstance(y_labels, str) else y_labels

        if len(data) == 1:
            data = [data[0]] * len(self._data)

        if len(y_labels) == 1:
            y_labels *= len(data)

        for y_label, axis in zip(y_labels, axes):
            self._axis_init(axis)
            axis.set_ylabel(y_label, fontsize=self._major)

        self._axis_plot_data(
            self._colours[1],
            self._hatches[1] if isinstance(self._hatches, tuple) else self._hatches,
            data,
            axes,
        )

        if label:
            self._labels = self._labels + [label] if self._labels is not None else [label]
            self.legend.remove()
            self.create_legend(**self._legend_kwargs)


class PlotImages(BasePlot):
    """
    Plots images as a grid

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
            data: ndarray,
            num_plots: int = 12,
            cmaps: str | list[str] | None = None,
            titles: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            ranges: tuple[float, float] | list[tuple[float, float]] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : (B,W,H) ndarray
            B image data of width W and height H
        num_plots : int, default = 12
            Maximum number of images to plot
        cmaps : str | list[str] | None, default = 'hot'
            Colour map(s) for each image
        titles: list[str] | None, default = None
            Titles for each image
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure
        ranges : tuple[float, float] | list[tuple[float, float]] | ndarray | None, default = None
            Lower and upper bounds for each image, default is minimum and maximum for each image

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._num_plots: int = min(num_plots, len(data))
        self._cmaps: list[str]
        self._ranges: list[tuple[float, float]] | ndarray
        self._data: ndarray

        if isinstance(cmaps, str):
            self._cmaps = [cmaps] * self._num_plots
        else:
            self._cmaps = cmaps or ['hot'] * self._num_plots

        if ranges is None:
            self._ranges = [(np.min(data), np.max(data))] * self._num_plots
        elif np.ndim(ranges) < 2:
            self._ranges = [ranges] * self._num_plots
        else:
            self._ranges = ranges

        super().__init__(data, labels=titles, fig_size=fig_size, **kwargs)

    def _axes_init(self) -> None:
        self.subplots(subplot_grid(self._num_plots), titles=self._labels)
        self._multi_param_func_calls('grid', self.axes.values(), [False])

    def _post_init(self) -> None:
        self._default_name = 'images'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        cmap: str
        range_: tuple[float, float] | ndarray
        data: ndarray
        axis: Axes

        for cmap, range_, data, axis in zip(
                self._cmaps,
                self._ranges,
                self._data,
                self.axes.values()):
            self.plots.append(axis.imshow(data, cmap=cmap, vmin=range_[0], vmax=range_[1]))
            axis.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

    def create_legend(self, *_, **__) -> None:
        """
        Confusion matrix should not have a legend
        """
        return


class PlotSaliency(PlotImages):
    """
    Plots the saliency and input for multiple saliency maps

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
            data: ndarray,
            saliency: ndarray,
            num_plots: int = 12,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : (W,H) ndarray
            Original input data of width W and height H
        saliency : (B,W,H) ndarray
            B sets of saliency maps of width W and height H
        num_plots : int, default = 12
            Maximum number of saliencies to plot
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to PlotImages
        """
        super().__init__(
            np.concat((data[np.newaxis], saliency)),
            num_plots=num_plots,
            cmaps=['hot'] + ['twilight'] * len(saliency),
            titles=['Input'] + [f'Dim {i + 1}' for i in range(len(saliency))],
            ranges=np.concat((
                [[0, np.max(data)]],
                np.min(saliency, axis=(1, 2)),
                np.max(saliency, axis=(1, 2)),
            )),
            fig_size=fig_size,
            **kwargs,
        )

    def _post_init(self) -> None:
        self._default_name = 'saliency'
