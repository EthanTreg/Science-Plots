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
            data_twin: list[ndarray] | ndarray | None = None,
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
        data_twin : list[(M) ndarray] | (M) ndarray | (B,M) ndarray | None, default = None
            B twin distributions with M data points to plot on each distribution

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
        self._data_twin: list[ndarray] | list[None] | ndarray = [None] * len(self._data) \
            if data_twin is None else data_twin

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

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        text: str
        range_: tuple[float, float]
        data: ndarray
        data_twin: ndarray
        axis: Axes

        for text, data, data_twin, axis in zip(
                self._texts,
                self._data,
                self._data_twin,
                self.axes.values()):
            range_ = (
                    min(np.min(data), np.min(data_twin)),
                    max(np.max(data), np.max(data_twin)),
                )
            axis.set_xscale('log' if self._log else 'linear')
            self.plot_hist(
                self._colours[0],
                data,
                axis,
                log=self._log,
                norm=self._norm,
                hatch=self._hatches[0] if isinstance(self._hatches, tuple) else self._hatches,
                range_=range_,
            )
            self.plot_hist(
                self._colours[1],
                data_twin,
                axis,
                log=self._log,
                norm=self._norm,
                hatch=self._hatches[1] if isinstance(self._hatches, tuple) else self._hatches,
                range_=(
                    min(np.min(data), np.min(data_twin)),
                    max(np.max(data), np.max(data_twin)),
                ),
            )

            if not self._y_axes:
                axis.tick_params(labelleft=False, left=False)

            if text:
                axis.add_artist(mpl.offsetbox.AnchoredText(
                    text,
                    loc=self._text_loc,
                    prop={'fontsize': utils.MINOR}
                ))


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
