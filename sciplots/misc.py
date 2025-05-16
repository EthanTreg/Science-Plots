"""
Miscellaneous plotting classes
"""
from typing import Any

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from sciplots import utils
from sciplots.base import BasePlot
from sciplots.utils import subplot_grid


class PlotImages(BasePlot):
    """
    Plots images as a grid

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
            num_plots: int = 12,
            cmaps: str | list[str] = 'hot',
            titles: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            ranges: tuple[float, float] | list[tuple[float, float]] | ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        data : list[(W,H) ndarray] | (B,W,H) ndarray
            B image data of width W and height H
        num_plots : int, default = 12
            Maximum number of images to plot
        cmaps : str | list[str], default = 'hot'
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
        self._titles: list[str] | None = titles
        self._ranges: list[tuple[float, float]] | ndarray
        self._data: ndarray

        if isinstance(cmaps, str):
            self._cmaps = [cmaps] * self._num_plots
        else:
            self._cmaps = cmaps

        if ranges is None:
            self._ranges = [(np.min(datum), np.max(datum)) for datum in data]
        elif np.ndim(ranges) < 2:
            self._ranges = [ranges] * self._num_plots
        else:
            self._ranges = ranges

        super().__init__(data, fig_size=fig_size, **kwargs)

    def _axes_init(self) -> None:
        self.subplots(subplot_grid(self._num_plots), titles=self._titles)
        utils.cast_func('grid', self.axes, args=[False] * self._num_plots)

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
            self.plots[axis].append(axis.imshow(data, cmap=cmap, vmin=range_[0], vmax=range_[1]))
            axis.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

    def create_legend(self, *_: Any, **__: Any) -> None:
        """
        Confusion matrix should not have a legend
        """
        return

    def set_axes_pad(self, **_: Any) -> None:
        """
        Images should not have axis padding
        """
        return


class PlotSaliency(PlotImages):
    """
    Plots the saliency and input for multiple saliency maps

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
            data: ndarray,
            saliency: ndarray,
            num_plots: int = 12,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any) -> None:
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
