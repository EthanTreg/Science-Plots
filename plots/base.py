"""
Base plotting class for other plots to build upon
"""
import os
import logging
from typing import Any

import numpy as np
import scienceplots  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure, FigureBase
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from plots import utils

plt.style.use(["science", "grid", 'no-latex'])


class BasePlot:
    """
    Base class for creating plots

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
    savefig(path, name='')
        Saves the plot to the specified path
    subfigs(subfigs, titles=None, x_labels=None, y_labels=None, **kwargs)
        Initialises sub figures
    subplots(subplots, titles=None, fig=None, **kwargs)
        Generates subplots within a figure or sub-figure
    create_legend(rows=1, loc='outside upper center')
        Plots the legend
    plot_density(colour, data, ranges, axis, hatch='', order=None, confidences=None, **kwargs)
        Plots a density contour plot
    plot_errors(colour, x_data, y_data, axis, error_region=False, x_error=None, y_error=None,
            **kwargs)
        Plots errors on a scatter plot
    plot_histogram(colour, data, axis, log=False, norm=False, orientation='vertical', hatch='',
            range_=None, **kwargs)
        Plots a histogram or density plot
    plot_param_pairs(colour, data, hatch='', ranges=None, markers=None, **kwargs)
        Plots a pair plot to compare the distributions and comparisons between parameters
    plot_residuals(colour, x_data, pred, target, axis, error=False, error_region=False,
            target_colour=None, uncertainty=None, major_axis=None)
        Plots comparison and residuals
    """
    def __init__(
            self,
            data: Any,
            density: bool = False,
            bins: int = 100,
            alpha: float = 0.6,
            alpha_2d: float = 0.4,
            title: str = '',
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            fig_size: tuple[float, float] = utils.RECTANGLE,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : Any
            Data for the plot
        density : bool, default = False
            If the plot should plot contours and interpolate histograms
        bins : int, default = 200
            Number of bins for histograms or density interpolation
        alpha : float, default = 0.6
            Alpha value for scatter points and hatching
        alpha_2d : float, default = 0.4
            Alpha value for areas such as error regions
        title : str, default = ''
            Title of the plot
        x_label : str, default = ''
            X-label of the plot
        y_label : str, default = ''
            Y-label of the plot
        labels : list[str] | None, default = None
            Labels for the data to plot the legend
        colours : list[str] | None, default = XKCD_COLORS
            Colours for the data
        fig_size : tuple[float, float], default = RECTANGLE
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to create_legend
        """
        self._density: bool = density
        self._bins: int = bins
        self._alpha: float = alpha
        self._alpha_2d: float = alpha_2d
        self._minor_tick_factor: float = 0.8
        self._marker_size: float = 100 * fig_size[0] / utils.RECTANGLE[0]
        self._major: float = utils.MAJOR * fig_size[0] / utils.RECTANGLE[0]
        self._minor: float = utils.MINOR * fig_size[0] / utils.RECTANGLE[0]
        self._default_name: str = 'base'
        self._labels: list[str] | None = labels
        self._colours: list[str] = colours or utils.COLOURS
        self._fig_size: tuple[float, float] = fig_size
        self._data: Any = data
        self._legend_axis: Axes | None = None
        self._legend_kwargs: dict[str, Any] = kwargs
        self.plots: list[Artist] = []
        self.axes: dict[int | str, Axes] | ndarray | Axes
        self.subfigs: ndarray | None = None
        self.fig: Figure = plt.figure(constrained_layout=True, figsize=self._fig_size)
        self.legend: Legend | None = None

        self.fig.suptitle(title, fontsize=self._major)
        self.fig.supxlabel(x_label, fontsize=self._major)
        self.fig.supylabel(y_label, fontsize=self._major)
        self._axes_init()
        self._post_init()
        self._plot_data()

        if self._labels is not None and self._labels[0]:
            self.create_legend(**self._legend_kwargs)

    @staticmethod
    def _multi_param_func_calls(
            func: str,
            objs: list[object] | ndarray,
            *args: list[Any] | None,
            kwargs: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
        """
        Calls a function for multiple objects with different arguments and keyword arguments

        Parameters
        ----------
        func : str
            Name of the object's function to call
        objs : list[object] | (N) ndarray
            List of N objects to apply the function to

        *args : list[Any] | None
            Arguments to pass to the function
        **kwargs : list[dict[str, Any]] | dict[str, Any] | None, default = None
            Optional keyword arguments to pass to the function
        """
        obj: object
        kwarg: dict[str, Any]
        arg: Any

        if len(args) != 0 and args[0] is None:
            return
        if len(args) == 0 or args[0] is None:
            args = [()] * len(objs)
        else:
            args = [tuple(arg) for arg in zip(*args)]

        if kwargs is None:
            kwargs = [{}] * len(objs)
        elif isinstance(kwargs, dict):
            kwargs = [kwargs] * len(objs)

        for obj, arg, kwarg in zip(objs, args, kwargs):
            getattr(obj, func)(*arg, **kwarg)

    def _axis_init(self, axis: Axes) -> None:
        """
        Initialises a provided axis

        Parameters
        ----------
        axis : Axes
            Axis to initialise
        """
        axis.tick_params(labelsize=self._minor)
        axis.tick_params(labelsize=self._minor * self._minor_tick_factor, which='minor')


    def _axes_init(self) -> None:
        """
        Initialises the axes
        """
        self.axes = self.fig.gca()
        self._axis_init(self.axes)

    def _post_init(self) -> None:
        """
        Performs any necessary post-initialisation tasks
        """

    def _plot_data(self) -> None:
        """
        Plots the data
        """

    def savefig(self, path: str, name: str = '', **kwargs: Any) -> None:
        """
        Saves the plot to the specified path

        Parameters
        ----------
        path : str
            Path to save the plot
        name : str, default = ''
            Name of the plot, if empty, default name for the plot will be used

        **kwargs
            Optional keyword arguments to pass to Figure.savefig
        """
        name = name or self._default_name
        name += '.png' if '.png' not in name else ''
        self.fig.savefig(os.path.join(path, name), **kwargs)

    def subfigures(
            self,
            subfigs: tuple[int, int],
            titles: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            **kwargs: Any) -> None:
        """
        Initialises sub figures

        Parameters
        ----------
        subfigs : tuple[int, int]
            Number of rows and columns for the sub figures
        titles : list[str] | None, default = None
            Title for each sub figure
        x_labels : list[str] | None, default = None
            X-label for each sub figure
        y_labels : list[str] | None, default = None
            Y-label for each sub figure

        **kwargs
            Optional arguments for the subfigures function
        """
        self.subfigs = self.fig.subfigures(*subfigs, **kwargs)
        self._multi_param_func_calls(
            'suptitle',
            self.subfigs.flatten(),
            titles,
            kwargs={'fontsize': self._major},
        )
        self._multi_param_func_calls(
            'supxlabel',
            self.subfigs.flatten(),
            x_labels,
            kwargs={'fontsize': self._major},
        )
        self._multi_param_func_calls(
            'supylabel',
            self.subfigs.flatten(),
            y_labels,
            kwargs={'fontsize': self._major},
        )

    def subplots(
            self,
            subplots: str | tuple[int, int] | list[list[int | str]] | ndarray,
            titles: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            fig: FigureBase | None = None,
            **kwargs: Any) -> None:
        """
        Generates subplots within a figure or sub-figure

        Parameters
        ----------
        subplots : str | tuple[int, int] | list[list[int | str]] | (R,C) ndarray
            Parameters for subplots or subplot_mosaic for R rows and C columns
        titles : list[str] | None, default = None
            Titles for each axis
        x_labels : list[str] | None, default = None
            Labels for the x-axis
        y_labels : list[str] | None, default = None
            Labels for the y-axis
        fig : FigureBase | None, default = self.fig
            Figure or sub-figure to add subplots to

        **kwargs
            Optional kwargs to pass to subplots or subplot_mosaic
        """
        fig = fig or self.fig

        if isinstance(subplots, tuple):
            self.axes = fig.subplots(*subplots, **kwargs)
        else:
            self.axes = fig.subplot_mosaic(subplots, **kwargs)

        self._multi_param_func_calls(
            'set_title',
            self.axes.flatten() if isinstance(self.axes, ndarray) else self.axes.values(),
            titles,
            kwargs={'fontsize': self._major},
        )
        self._multi_param_func_calls(
            'tick_params',
            self.axes.flatten() if isinstance(self.axes, ndarray) else self.axes.values(),
            kwargs={'labelsize': self._minor},
        )
        self._multi_param_func_calls(
            'tick_params',
            self.axes.flatten() if isinstance(self.axes, ndarray) else self.axes.values(),
            kwargs={'labelsize': self._minor * self._minor_tick_factor, 'which': 'minor'},
        )

        if isinstance(self.axes, ndarray) and x_labels:
            self._multi_param_func_calls(
               'set_xlabel',
                self.axes[-1],
                x_labels,
                kwargs={'fontsize': self._major},
            )

        if isinstance(self.axes, ndarray) and y_labels:
            self._multi_param_func_calls(
               'set_ylabel',
                self.axes[:, 0],
                y_labels,
                kwargs={'fontsize': self._major},
            )

    def create_legend(
            self,
            axis: bool = False,
            rows: int = 1,
            loc: str | tuple[float, float] = 'outside upper center') -> None:
        """
        Plots the legend

        Parameters
        ----------
        axis : bool, default = False
            Whether to plot the legend on the axes or the figure
        rows : int, default = 1
            Number of rows for the legend
        loc : str | tuple[float, float], default = 'outside upper center'
            Location to place the legend
        """
        cols: int = np.ceil(len(self._labels) / rows)
        fig_size: float = self.fig.get_size_inches()[0] * self.fig.dpi
        handles: list[tuple[Artist, ...]]
        legend_range: ndarray
        handle: ndarray | Artist
        artist: Figure | Axes

        # Generate legend handles
        self.plots = [handle.legend_elements()[0][0] if isinstance(handle, ContourSet) else handle
                      for handle in self.plots]
        handles = [tuple(handle) for handle in np.array_split(self.plots, len(self._labels))]

        if axis and self._legend_axis is None:
            match self.axes:
                case np.ndarray():
                    artist = self.axes.flatten()[0]
                case dict():
                    artist = list(self.axes.values())[0]
                case Axes():
                    artist = self.axes
        elif axis:
            artist = self._legend_axis
        else:
            artist = self.fig

        # Create legend
        self.legend = artist.legend(
            handles,
            self._labels,
            fancybox=False,
            ncol=cols,
            fontsize=self._major,
            borderaxespad=0.2 * utils.RECTANGLE[1] / self._fig_size[1],
            loc=loc,
            handler_map={tuple: utils.UniqueHandlerTuple(ndivide=None)}
        )
        legend_range = np.array(self.legend.get_window_extent())[:, 0]

        # Recreate legend if it overflows the figure with more rows
        if legend_range[0] < 0:
            self.legend.remove()
            rows = np.ceil((legend_range[1] - legend_range[0]) * rows / fig_size)
            self.create_legend(axis=axis, rows=rows, loc=loc)

        # Update handles to remove transparency if there isn't any hatching and set point size
        for handle in self.legend.legend_handles:
            if not hasattr(handle, 'get_hatch') or handle.get_hatch() is None:
                handle.set_alpha(1)

            if isinstance(handle, PathCollection):
                handle.set_sizes([100 * self._fig_size[0] / utils.RECTANGLE[0]])

            if isinstance(handle, Line2D):
                handle.set_linewidth(2)

    def plot_density(
            self,
            colour: str,
            data: ndarray,
            ranges: ndarray,
            axis: Axes,
            hatch: str = '',
            order: list[int] | None = None,
            confidences: list[float] | None = None,
            **kwargs: Any) -> None:
        """
        Plots a density contour plot

        Parameters
        ---------
        colour : str
            Colour of the contour
        data : (N,2) ndarray
            N (x,y) data points to generate density contour for
        ranges : (2,2) ndarray
            Min and max values for the x and y axes
        axis : Axes
            Axis to add density contour
        hatch : str, default = ''
            Hatching pattern for the contour
        order : list[int] | None, default = None
            Order of the axes, only required for 3D plots
        confidences : list[float] | None, default = [0.68]
            List of confidence values to plot contours for, starting with the lowest confidence

        **kwargs
            Optional kwargs to pass to Axes.contour and Axes.contourf
        """
        total: float
        levels: list[float]
        logger: logging.Logger = logging.getLogger(__name__)
        contour: ndarray
        grid: ndarray = np.mgrid[
                        ranges[0, 0]:ranges[0, 1]:self._bins * 1j,
                        ranges[1, 0]:ranges[1, 1]:self._bins * 1j,
                        ]
        kernel: gaussian_kde

        try:
            kernel = gaussian_kde(data.swapaxes(0, 1))
        except np.linalg.LinAlgError:
            logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.reshape(kernel(grid.reshape(2, -1)).T, (self._bins, self._bins))
        total = np.sum(contour)
        levels = [np.max(contour)]

        if confidences is None:
            confidences = [0.68]

        for confidence in confidences:
            levels.insert(0, utils.contour_sig(total * confidence, contour))

        if levels[-1] == 0:
            logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.concatenate((grid, contour[np.newaxis]), axis=0)

        if order is not None:
            contour = contour[order]

        self.plots.append(axis.contourf(
            *contour,
            levels,
            alpha=self._alpha_2d,
            colors=colour,
            hatches=[hatch],
            **kwargs,
        ))
        self.plots[-1]._hatch_color = colors.to_rgba(colour, self._alpha)
        self.plots.append(axis.contour(*contour, levels, colors=colour, **kwargs))

    def plot_errors(
            self,
            colour: str,
            x_data: ndarray,
            y_data: ndarray,
            axis: Axes,
            error_region: bool = False,
            marker: str = 'x',
            x_error: ndarray | None = None,
            y_error: ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Plots errors on a scatter plot

        Parameters
        ----------
        colour : str
            Colour of the data and errors
        x_data : (N) ndarray
            N x-data points
        y_data : (N) ndarray
            N y-data points
        axis : Axes
            Axis to plot the data
        marker : str, default = 'x'
            Marker style for the data, if '', plot will be used
        error_region : bool, default = False
            If the errors should be error bars or a highlighted region, highlighted region only
            supports y_errors
        x_error : (N) ndarray | None, default = None
            N x-errors
        y_error : (N) ndarray | None, default = None
            N y-errors

        **kwargs
            Optional keyword arguments to pass to Axes.scatter and Axes.errorbar
        """
        if marker:
            self.plots.append(axis.scatter(
                x_data,
                y_data,
                color=colour,
                marker=marker,
                s=self._marker_size,
                alpha=self._alpha,
                **kwargs,
            ))
        else:
            self.plots.append(axis.plot(
                x_data,
                y_data,
                alpha=self._alpha,
                color=colour,
                **kwargs,
            )[0])

        if error_region and y_error is not None:
            self.plots.append(axis.fill_between(
                x_data,
                y_data + y_error,
                y_data - y_error,
                color=colour,
                alpha=self._alpha_2d
            ))
        elif x_error is not None or y_error is not None:
            self.plots.append(axis.errorbar(
                x_data,
                y_data,
                yerr=y_error,
                xerr=x_error,
                capsize=5,
                alpha=self._alpha,
                color=colour,
                **kwargs,
            )[0])

    def plot_grid(
            self,
            matrix: ndarray,
            axis: Axes,
            diverge: bool = False,
            precision: int = 3,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            range_: tuple[float, float] | None = None) -> None:
        """
        Plots a grid of values using an image plot

        Parameters
        ----------
        matrix : (N,M) ndarray
            2D array of values to plot
        axis : Axes
            Axis to plot on
        diverge : bool, default = False
            If the colour map should diverge or be sequential
        precision : int, default = 3
            Number of decimal places to display
        x_labels : list[str] | None, default = None
            Labels for the x-axis
        y_labels : list[str] | None, default = None
            Labels for the y-axis
        range_ : tuple[float, float] | None, default = (min(matrix), max(matrix))
            Range of values for colouring, if None, the range will be the minimum and maximum of the
            matrix
        """
        i: int
        j: int
        value: float
        colour: str
        range_ = range_ or (np.min(matrix), np.max(matrix))
        self.plots.append(axis.imshow(
            matrix,
            vmin=range_[0],
            vmax=range_[1],
            cmap='PiYG' if diverge else 'Blues',
        ))
        axis.grid(False)

        if x_labels:
            axis.set_xticks(np.arange(len(x_labels)), x_labels, fontsize=self._minor)

        if y_labels:
            axis.set_yticks(
                np.arange(len(y_labels)),
                y_labels,
                rotation=90,
                va='center',
                fontsize=self._minor,
            )

        for (i, j), value in zip(np.ndindex(matrix.shape), matrix.flatten()):
            if diverge:
                colour = 'w' if (np.abs(value) >
                                 np.abs(range_[0] + 7 * (range_[1] - range_[0]) / 8)) else 'k'
            else:
                colour = 'w' if value > range_[0] + (range_[1] - range_[0]) / 2 else 'k'

            axis.text(
                j, i, f'{value:.{precision}f}',
                ha='center',
                va='center',
                color=colour,
                fontsize=self._minor,
            )

    def plot_hist(
            self,
            colour: str,
            data: ndarray,
            axis: Axes,
            log: bool = False,
            norm: bool = False,
            orientation: str = 'vertical',
            hatch: str = '',
            range_: tuple[float, float] | None = None,
            **kwargs: Any) -> None:
        """
        Plots a histogram or density plot

        Parameters
        ----------
        colour : str
            Colour of the histogram or density plot
        data : (N) ndarray
            Data to plot for N data points
        axis : Axes
            Axis to plot on
        log : bool, default = False
            If data should be plotted on a log scale, expects linear data
        norm : bool, default = False
            If the histogram or density plot should be normalised to a maximum height of 1
        orientation : {'vertical', 'horizontal'}, default = 'vertical'
            Orientation of the histogram or density plot
        hatch : str, default = ''
            Hatching of the histogram or density plot
        range_ : tuple[float, float], default = None
            x-axis data range

        **kwargs
            Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if
            self._density is True, else to Axes.hist
        """
        edge_alpha: float = 0.8
        x_data: ndarray
        y_data: ndarray
        kernel: gaussian_kde
        axis.ticklabel_format(axis='y', scilimits=(-2, 2))

        if orientation not in {'vertical', 'horizontal'}:
            raise ValueError(f"Orientation ({orientation}) must be 'vertical' or 'horizontal'")

        if log:
            axis.set_xscale('log')

        if len(np.unique(data)) == 1:
            norm = True

        if range_ is None:
            range_ = (np.min(data), np.max(data))

        if self._density and len(np.unique(data)) > 1:
            kernel = gaussian_kde(np.log10(data) if log else data)
            x_data = np.linspace(*(np.log10(range_) if log else range_), self._bins)
            y_data = kernel(x_data)
            x_data = 10 ** x_data if log else x_data

            if norm:
                y_data /= np.max(y_data)

            if orientation == 'vertical':
                self.plots.append(axis.plot(
                    x_data,
                    y_data,
                    color=colour,
                    **kwargs,
                )[0])
                self.plots.append(axis.fill_between(
                    x_data,
                    y_data,
                    hatch=hatch,
                    facecolor=(colour, self._alpha_2d),
                    edgecolors=(colour, self._alpha),
                    **kwargs,
                ))
            else:
                self.plots.append(axis.plot(
                    y_data,
                    x_data,
                    color=colour,
                    **kwargs,
                )[0])
                self.plots.append(axis.fill_betweenx(
                    x_data,
                    y_data,
                    hatch=hatch,
                    facecolor=(colour, self._alpha_2d),
                    edgecolor=(colour, self._alpha),
                    **kwargs,
                ))
        elif norm:
            y_data, x_data = np.histogram(
                data,
                np.logspace(*np.log10(range_), self._bins) if log else self._bins,
            )
            y_data = y_data / np.max(y_data)
            (axis.bar if orientation == 'vertical' else axis.barh)(
                x_data[:-1],
                y_data,
                np.diff(x_data),
                align='edge',
                color=colour,
                alpha=self._alpha_2d,
                **kwargs,
            )
        else:
            self.plots.append(axis.hist(
                data,
                lw=2,
                bins=np.logspace(*np.log10(range_), self._bins) if log else self._bins,
                hatch=hatch,
                histtype='step',
                alpha=edge_alpha,
                orientation=orientation,
                facecolor=(colour, self._alpha_2d),
                edgecolor=(colour, edge_alpha),
                range=range_,
                **kwargs,
            )[-1][0])
            self.plots.append(axis.hist(
                data,
                lw=2,
                bins=np.logspace(*np.log10(range_), self._bins) if log else self._bins,
                hatch=hatch,
                histtype='stepfilled',
                alpha=self._alpha_2d,
                orientation=orientation,
                facecolor=(colour, self._alpha_2d),
                edgecolor=(colour, edge_alpha),
                range=range_,
                **kwargs,
            )[-1][0])

    def plot_param_pairs(
            self,
            colour: str,
            data: ndarray,
            norm: bool = False,
            hatch: str = '',
            ranges: ndarray | None = None,
            markers: ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Plots a pair plot to compare the distributions and comparisons between parameters

        Parameters
        ----------
        colour : str
            Colour of the data
        data : (N,L) ndarray
            Data to plot parameter pairs for N data points and L parameters
        hatch : str = ''
            Hatching of the histograms or density plots and contours
        ranges : (L,2) ndarray, default = None
            Ranges for L parameters, required if using kwargs to plot densities
        markers : (N) ndarray | None = None
            Markers for scatter plots for N data points

        **kwargs
            Optional keyword arguments passed to _plot_histogram and _plot_density
        """
        assert isinstance(self.axes, ndarray)
        i: int
        j: int
        x_data: ndarray
        y_data: ndarray
        x_range: ndarray
        y_range: ndarray
        axes_row: ndarray
        axis: Axes

        if markers is None:
            markers = np.array(['.'] * len(data))

        data = np.swapaxes(data, 0, 1)

        # Loop through each subplot
        for i, (axes_row, y_data, y_range) in enumerate(zip(self.axes, data, ranges)):
            for j, (axis, x_data, x_range) in enumerate(zip(axes_row, data, ranges)):
                # Share y-axis for all scatter plots
                if i != j:
                    axis.sharey(axes_row[0])

                # Set number of ticks
                axis.locator_params(axis='x', nbins=3)
                axis.locator_params(axis='y', nbins=3)

                # Hide ticks for plots that aren't in the first column or bottom row
                if j != 0 or j == i:
                    axis.tick_params(labelleft=False, left=False)

                if i != self.axes.shape[0] - 1:
                    axis.tick_params(labelbottom=False, bottom=False)

                if j < i:
                    axis.set_xlim(x_range)
                    axis.set_ylim(y_range)

                # Plot scatter plots & histograms
                if i == j:
                    self.plot_hist(
                        colour,
                        x_data,
                        axis,
                        norm=norm,
                        hatch=hatch,
                        range_=x_range,
                        **kwargs,
                    )
                elif j < i:
                    for marker in np.unique(markers):
                        self.plots.append(axis.scatter(
                            x_data[marker == markers][:utils.SCATTER_NUM],
                            y_data[marker == markers][:utils.SCATTER_NUM],
                            s=self._marker_size,
                            alpha=self._alpha,
                            color=colour,
                            marker=marker,
                        ))

                    if self._density and len(np.unique(x_data)) > 1 and len(np.unique(y_data)) > 1:
                        self.plot_density(
                            colour,
                            np.stack((x_data, y_data), axis=1),
                            np.array((x_range, y_range)),
                            axis=axis,
                            hatch=hatch,
                            **kwargs,
                        )
                else:
                    axis.set_visible(False)

    def plot_residuals(
            self,
            colour: str,
            x_data: ndarray,
            pred: ndarray,
            target: ndarray,
            axis: Axes,
            error: bool = False,
            error_region: bool = False,
            target_colour: str | None = None,
            uncertainty: ndarray | None = None,
            major_axis: Axes | None = None,
            **kwargs: Any) -> Axes:
        """
        Plots comparison and residuals

        Parameters
        ----------
        colour : str
            Colour of the data
        x_data : (N) ndarray
            X-axis values for N data points
        pred : (N) ndarray
            Predicted values for N data points
        target : (N) ndarray
            Target values for N data points
        axis : Axes
            Axis to plot on
        error : bool, default = False,
            If the residuals should be plotted as residuals or errors
        error_region : bool, default = False
            If the errors should be error bars or a highlighted region
        target_colour : str | None, default = None
            Colour of the target data, if None, colour will be used
        uncertainty : (N) ndarray | None, default = None
            Uncertainties in the y predictions for N data points
        major_axis : Axes | None, default = None
            Major axis of the plot with the comparison
        **kwargs
            Optional keyword arguments to pass to self.plot_errors

        Returns
        -------
        Axes
            Major axis of the plot with the comparison
        """
        major_axis = major_axis or make_axes_locatable(axis).append_axes('top', size='150%', pad=0)
        self.plot_errors(
            colour,
            x_data,
            (pred - target) / target if error else pred - target,
            axis,
            error_region=error_region,
            y_error=uncertainty / target if error else uncertainty,
            **kwargs,
        )
        self.plot_errors(
            colour,
            x_data,
            pred,
            major_axis,
            error_region=error_region,
            y_error=uncertainty,
            **kwargs,
        )
        major_axis.plot(x_data, target, color=target_colour or colour)

        axis.set_ylabel('Error' if error else 'Residual', fontsize=self._major)
        axis.hlines(0, xmin=np.min(x_data), xmax=np.max(x_data), color='k')
        major_axis.tick_params(axis='y', labelsize=self._minor)
        major_axis.set_xticks([])
        return major_axis
