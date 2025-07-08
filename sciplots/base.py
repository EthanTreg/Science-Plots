"""
Base plotting class for other sciplots to build upon
"""
import os
import logging
from warnings import warn
from typing import Any, Type

import numpy as np
import scienceplots  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.patches import Polygon
from matplotlib.colors import XKCD_COLORS
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure, FigureBase
from matplotlib.container import Container, BarContainer, ErrorbarContainer
from matplotlib.collections import (
    Collection,
    LineCollection,
    PathCollection,
    FillBetweenPolyCollection,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from sciplots import utils

plt.style.use(["science", "grid", 'no-latex'])


class BasePlot:
    """
    Base class for creating sciplots

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
    legend_axis : Axes | None, default = None
        Axis to plot the legend on if argument axis is True in create_legend method, if None and
        axis argument is True, first axis will be used
    legend : Legend | None, default = None
        Plot legend

    Methods
    -------
    cast(func, objs, obj_first=True, kwargs_unique=False, args=None, kwargs=None) -> list[Any]
        Casts args and kwargs to a given function for a set of objects
    savefig(path, name='', **kwargs)
        Saves the plot to the specified path
    subfigs(subfigs, titles=None, x_labels=None, y_labels=None, **kwargs)
        Initialises sub figures
    subplots(subplots, titles=None, x_labels=None, y_labels=None, fig=None, **kwargs)
        Generates subplots within a figure or sub-figure
    create_legend(axis=False, rows=1, cols=0, loc='outside upper center')
        Plots the legend
    set_axes_pad(pad=0)
        Sets the padding on all plot axes
    plot_density(colour, data, ranges, axis, label='', hatch='', order=None, confidences=None,
            **kwargs)
        Plots a density contour plot
    plot_errors(colour, x_data, y_data, axis, label='', style='x', x_error=None, y_error=None,
            **kwargs)
        Plots errors on a scatter plot
    plot_grid(matrix, axis, diverge=False, precision=3, x_labels=None, y_labels=None, range_=None)
        Plots a grid of values using an image plot
    plot_hist(colour, data, axis, log=False, norm=False, label='', orientation='vertical', hatch='',
            range_=None, **kwargs)
        Plots a histogram or density plot
    plot_param_pairs(colour, data, norm=False, label='', hatch='', ranges=None, markers=None,
            **kwargs)
        Plots a pair plot to compare the distributions and comparisons between parameters
    plot_residuals(colour, x_data, pred, target, axis, error=False, label='', target_colour=None,
            uncertainty=None, major_axis=None, **kwargs)
        Plots comparison and residuals
    """
    _error_region: bool = False
    _scatter_num: int = 1000
    _pad: float = 0
    _major: float = 24
    _minor: float = 20
    _cap_size: float = 5
    _alpha_2d: float = 0.4
    _line_width: float = 2
    _eline_width: float = 2
    _alpha_line: float = 0.9
    _marker_size: float = 50
    _alpha_marker: float = 0.6
    _minor_tick_factor: float = 0.8
    _handle_priority: list[Type[Artist]] = [
        FillBetweenPolyCollection,
        BarContainer,
        Polygon,
        Line2D,
        LineCollection,
        PathCollection,
    ]
    _logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
            self,
            data: Any,
            density: bool = False,
            bins: int = 100,
            title: str = '',
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | ndarray | None = None,
            colours: list[str] | ndarray | None = None,
            fig_size: tuple[float, float] = utils.RECTANGLE,
            **kwargs: Any) -> None:
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
        labels : list[str] | ndarray | None, default = None
            Labels for the data to plot the legend
        colours : list[str] | ndarray | None, default = XKCD_COLORS
            Colours for the data
        fig_size : tuple[float, float], default = RECTANGLE
            Size of the figure in hundreds of pixels

        **kwargs
            Optional keyword arguments to pass to create_legend
        """
        scale: float = fig_size[1] / utils.RECTANGLE[1]
        self._density: bool = density
        self._bins: int = bins
        self._labels: list[str] = labels.tolist() if isinstance(labels, ndarray) else labels or ['']
        self._colours: list[str] = colours.tolist() if isinstance(colours, ndarray) else \
            colours or list(XKCD_COLORS.values())[::-1]
        self._data: Any = data
        self._legend_kwargs: dict[str, Any] = kwargs
        self.plots: dict[Axes, list[Artist | Container]] = {}
        self.axes: dict[int | str, Axes] | ndarray | Axes
        self.subfigs: ndarray | None = None
        self.fig: Figure = plt.figure(
            constrained_layout=True,
            figsize=(fig_size[0] / scale, fig_size[1] / scale),
            dpi=100 * scale,
        )
        self.legend_axis: Axes | None = None
        self.legend: Legend | None = None

        # Generation of the plot
        self._process_kwargs(self._legend_kwargs)
        self.fig.suptitle(title, fontsize=self._major)
        self.fig.supxlabel(x_label, fontsize=self._major)
        self.fig.supylabel(y_label, fontsize=self._major)
        self._axes_init()
        self._post_init()
        self._update_plots_dict()
        self._plot_data()
        self.set_axes_pad(pad=self._pad)

        if self._labels[0]:
            self.create_legend(**self._legend_kwargs)

    @staticmethod
    def _data_length_normalise(
            x_data: list[ndarray] | ndarray,
            lists: list[list[Any] | Any] | None = None,
            data: list[list[ndarray] | ndarray | None] | None = None) -> tuple[
        list[ndarray] | ndarray,
        list[list[Any]],
        list[list[ndarray] | list[None] | ndarray]]:
        """
        Normalises the length and format of plot data

        Parameters
        ----------
        x_data : list[ndarray] | ndarray | ndarray
            Primary data as list of B sets of x-values with ndarray shape of N, a (N) ndarray, or a
            (B,N) ndarray, if x_data has a length of 1 or has shape (N), then B will be set to the
            length of the first element in data if not None
        lists : list[list[Any] | Any] | None, default = None
            Additional data paired to the number B sets of data
        data : list[list[ndarray] | ndarray | None] | None, default = None
            Additional data, each a list of M sets of values with ndarray shape of N, a (N) ndarray,
            or a (M,N) ndarray, if the data has a length of 1 or has shape (N), then M will be set
            to the length of x_data, if not None

        Returns
        -------
        tuple[list[ndarray] | ndarray, list[list[Any]], list[list[ndarray | list[None] | ndarray]
            List of B sets of x_data with ndarray shape N, or a (B,N) ndarray; lists, each with
            length B; and additional data, each a list of B sets of data with ndarray shape N, or a
            (B,N) ndarray, if not None
        """
        x_data = [x_data] if np.ndim(x_data[0]) < 1 else x_data
        data = [datum if datum is None else
                [datum] if np.ndim(datum[0]) < 1 else datum for datum in data] \
            if data is not None else None

        if data is not None and len(x_data) == 1:
            x_data = [x_data[0]] * len(data[0])

        for i, datum in enumerate(data) if data is not None else []:
            if datum is None:
                data[i] = [None] * len(x_data)
            else:
                datum = [datum] if np.ndim(datum[0]) < 1 else datum
                data[i] = [datum[0]] * len(x_data) if len(datum) == 1 else datum

        for i, list_ in enumerate(lists) if lists is not None else []:
            if not isinstance(list_, list):
                lists[i] = [list_] * len(x_data)

        return x_data, lists, data

    @staticmethod
    def _unique_objects(func: callable, objects: ndarray) -> list[object]:
        """
        Returns objects if the return from the function is not the same for all objects

        Parameters
        ----------
        func : callable
            Function to call for each object
        objects : (N) ndarray
            N objects to see if they are all the same or not

        Returns
        -------
        list[object]
            List of objects if they are not all the same, else empty list
        """
        return objects.tolist() if len(np.unique(
            np.array([func(b) for b in objects], dtype=str)
        )) > 1 else []

    def _twin_axes(
            self,
            x_axis: bool = True,
            labels: str | list[str] = '') -> dict[int | str, Axes] | ndarray | Axes:
        """
        Gets the twin axes for either the x-axis or y-axis

        Parameters
        ----------
        x_axis : bool, default = True
            If the twin axis should be x or y
        labels : str | list[str], default = ''
            Labels for the x or y axes, if it is a string, all axes will have the same label

        Returns
        -------
        dict[int | str, Axes] | ndarray | Axes
            Twin axes with the same form as self.axes
        """
        axes: dict[int | str, Axes] | ndarray | Axes = utils.cast_func(
            'twinx' if x_axis else 'twiny',
            [self.axes] if isinstance(self.axes, Axes) else self.axes,
        )
        axes = axes[0] if isinstance(self.axes, Axes) else axes
        utils.cast_func(
            'tick_params',
            [axes] if isinstance(axes, Axes) else axes,
            kwargs={'labelsize': self._minor}
        )

        if labels:
            utils.cast_func(
                'set_ylabel' if x_axis else 'set_xlabel',
                [axes] if isinstance(axes, Axes) else axes,
                args=[labels] if isinstance(labels, str) else labels,
                kwargs={'fontsize': self._major},
            )

        self._update_plots_dict(axes)
        return axes

    def _patch_ranges(self, patch: Artist) -> tuple[ndarray, ndarray] | None:
        """
        Calculates the data range for the given patch

        Parameters
        ----------
        patch : Artist
            Patch to calculate the data range for

        Returns
        -------
        tuple[(2) ndarray, (2) ndarray] | None
            Minimum and maximum for x and y axes if the patch is known and has data
        """
        idx: int
        data: list[tuple[ndarray, ndarray]] | list[ndarray] | ndarray
        datum: ndarray
        widths: ndarray

        match patch:
            case Line2D():
                assert isinstance(patch, Line2D)
                data = np.stack((
                    np.arange(len(x_data))
                    if (x_data := patch.get_xdata()).dtype.type in {np.str_, np.object_} else
                    x_data,
                    np.arange(len(y_data))
                    if (y_data := patch.get_ydata()).dtype.type in {np.str_, np.object_} else
                    y_data,
                ), axis=-1)
            case Polygon():
                assert isinstance(patch, Polygon)
                data = patch.get_xy()
            case PathCollection():
                assert isinstance(patch, PathCollection)
                data = patch.get_offsets().data
            case LineCollection():
                assert isinstance(patch, LineCollection)
                data = np.concat([datum for datum in patch.get_segments() if np.ndim(datum) > 1])
            case Collection() | QuadContourSet():
                assert isinstance(patch, (Collection, QuadContourSet))
                data = patch.get_paths()[0].vertices
            case BarContainer():
                assert isinstance(patch, BarContainer)
                idx = patch.orientation == 'vertical'
                widths = np.array([
                    rect.get_width() if idx else rect.get_height() for rect in patch
                ])
                data = np.array([(
                    rect.get_xy()[int(~idx)],
                    rect.get_height() if idx else rect.get_width(),
                ) for rect in patch])
                data = np.concat((data, np.stack((
                    data[:, 0] + widths,
                    np.zeros_like(widths),
                ), axis=-1)))
                data = data[:, ::(1 if idx else -1)]
            case Container():
                data = [range_ for child in patch.get_children()
                        if (range_ := self._patch_ranges(child)) is not None]

                if not data:
                    return None

                data = np.concat(data)
            case Text():
                return None
            case _:
                self._logger.warning(f'Unknown plot type ({patch.__class__}), skipping calculation '
                                     f'of axis padding for this plot type')
                return None
        return np.min(data, axis=0), np.max(data, axis=0)

    def _process_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Sets any attributes found in kwargs and removes them from kwargs

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments to process
        """
        key: str
        value: Any
        class_kwargs: set[str] = {
            'error_region',
            'scatter_num,' 
            'pad',
            'major',
            'minor',
            'cap_size',
            'alpha_2d',
            'line_width',
            'eline_width',
            'alpha_line',
            'marker_size',
            'alpha_marker',
            'minor_tick_factor',
        }

        for key, value in list(kwargs.items()):
            if key in class_kwargs and hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
                del kwargs[key]

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

    def _update_plots_dict(
            self,
            axes: dict[int | str, Axes] | ndarray | Axes | None = None) -> None:
        """
        Adds all axes to the plots dictionary

        Parameters
        ----------
        axes : dict[int | str, Axes] | ndarray | Axes | None, default = None
            Axes to add to the plots dictionary, if None, self.axes will be used
        """
        axes = self.axes if axes is None else axes

        match axes:
            case Axes():
                assert isinstance(axes, Axes)
                self.plots[axes] = []
            case ndarray():
                assert isinstance(axes, ndarray)
                self.plots = self.plots | {axis: [] for axis in axes.flatten()}
            case dict():
                assert isinstance(axes, dict)
                self.plots = self.plots | {axis: [] for axis in axes.values()}

    def _post_init(self) -> None:
        """
        Performs any necessary post-initialisation tasks
        """

    def _plot_data(self) -> None:
        """
        Plots the data
        """

    def _axis_data_ranges(self, axis: Axes, pad: float = 0) -> ndarray | None:
        """
        Gets the range of all data on the given axis

        Returns
        -------
        (N, 2, 2) ndarray | None
            N sets of data minimum and maximum values for each axis or None if there is no data
        """
        range_: tuple[ndarray, ndarray] | None
        ranges: list[tuple[ndarray, ndarray]] | ndarray = []
        plot: Artist

        for plot in self.plots[axis]:
            if range_ := self._patch_ranges(plot):
                ranges.append(range_)

        if len(ranges) > 0:
            ranges = np.array([np.min(ranges, axis=(0, 1)), np.max(ranges, axis=(0, 1))])
            ranges += np.stack((
                ranges[:, 0] if axis.get_xscale() == 'log' else
                [np.max(np.abs(ranges), axis=0)[0]] * 2,
                ranges[:, 1] if axis.get_yscale() == 'log' else
                [np.max(np.abs(ranges), axis=0)[1]] * 2,
            ), axis=1) * np.array([[-pad], [pad]])
            return ranges
        return None

    def _label_handles(self) -> dict[str, list[Artist]]:
        """
        Gets the labels and handles for the legend

        Returns
        -------
        dict[str, list[Artist]]
            Labels with their respective handles
        """
        label: str
        labels: list[str]
        handles: list[Artist] = []
        label_handles: dict[str, list[Artist] | tuple[Artist, ...]] = {}
        plot_type: Type[Artist]
        idxs: ndarray
        plots: ndarray
        plot: Artist

        # Get unique artists
        plots = np.array([[]] + sum(self.plots.values(), []), dtype=object)[1:]
        idxs = np.unique([plot.__class__.__name__ for plot in plots], return_index=True)[1]

        # Get unique artists of the same type if they have a label
        for plot_type in plots[idxs]:
            idxs = np.array([isinstance(plot, plot_type.__class__) and
                             plot.get_label() != '' and
                             plot.get_label()[0] != '_' for plot in plots])

            if not idxs.any():
                continue

            match plot_type:
                case Line2D() | LineCollection():
                    handles += self._unique_objects(lambda x: x.get_linestyle(), plots[idxs])
                case Polygon() | FillBetweenPolyCollection() | PathCollection() | QuadContourSet():
                    handles += self._unique_objects(lambda x: x.get_hatch(), plots[idxs])
                case BarContainer():
                    handles += self._unique_objects(
                        lambda x: x.get_children()[0].get_hatch(),
                        plots[idxs],
                    )
                case ErrorbarContainer():
                    handles += self._unique_objects(
                        lambda x: x.get_children()[0].get_linestyle(),
                        plots[idxs],
                    )
                case _:
                    self._logger.warning(f'Unknown plot type ({plot_type.__class__.__name__}), '
                                         f'skipping handles for this plot type')

        labels = [handle.get_label() for handle in handles]

        # Get all handles for each label
        for label in self._labels:
            if not label:
                continue

            label_handles[label] = np.array(handles)[np.isin(labels, label)].tolist()

        # If any label doesn't have a handle, get the first artist type depending on priority
        for label, handles in label_handles.items():
            for plot_type in self._handle_priority:
                if len(handles) != 0:
                    break

                idxs = np.array([
                    isinstance(plot, plot_type) and plot.get_label() == label for plot in plots
                ])
                handles += plots[idxs].tolist()

            if len(handles) == 0:
                handles += plots[[plot.get_label() == label for plot in plots]].tolist()[:1]

        return label_handles

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
        name = name or self.__class__.__name__.lower().replace('plot', '')
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
        utils.cast_func(
            'suptitle',
            self.subfigs,
            args=titles or [''],
            kwargs={'fontsize': self._major},
        )
        utils.cast_func(
            'supxlabel',
            self.subfigs,
            args=x_labels or [''],
            kwargs={'fontsize': self._major},
        )
        utils.cast_func(
            'supylabel',
            self.subfigs,
            args=y_labels or [''],
            kwargs={'fontsize': self._major},
        )

    def subplots(
            self,
            subplots: int | str | tuple[int, int] | list[list[int | str]] | ndarray,
            borders: bool = False,
            titles: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            fig: FigureBase | None = None,
            **kwargs: Any) -> None:
        """
        Generates subplots within a figure or sub-figure

        Parameters
        ----------
        subplots : int | str | tuple[int, int] | list[list[int | str]] | (R,C) ndarray
            Parameters for subplots or subplot_mosaic for R rows and C columns
        borders : bool, default = False
            If axis labels should only apply to the far left and bottom axis
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
        axes: dict[int | str, Axes] | ndarray
        fig = fig or self.fig

        if isinstance(subplots, int):
            subplots = utils.subplot_grid(subplots)

        if isinstance(subplots, tuple):
            self.axes = fig.subplots(*subplots, **kwargs)
        else:
            self.axes = fig.subplot_mosaic(subplots, **kwargs)

        utils.cast_func(
            'set_title',
            self.axes,
            args=titles or [''],
            kwargs={'fontsize': self._major},
        )
        utils.cast_func('tick_params', self.axes, kwargs={'labelsize': self._minor})
        utils.cast_func(
            'tick_params',
            self.axes,
            kwargs={'labelsize': self._minor * self._minor_tick_factor, 'which': 'minor'},
        )

        if x_labels:
            axes = self.axes[-1] if isinstance(self.axes, ndarray) else (
                np.array(list(self.axes.values()))[np.unique(subplots[-1])])
            utils.cast_func(
                'set_xlabel',
                axes if borders else self.axes,
                args=x_labels,
                kwargs={'fontsize': self._major},
            )

        if y_labels:
            axes = self.axes[:, 0] if isinstance(self.axes, ndarray) else (
                np.array(list(self.axes.values()))[np.unique(subplots[:, 0])])
            utils.cast_func(
                'set_ylabel',
                axes if borders else self.axes,
                args=y_labels,
                kwargs={'fontsize': self._major},
            )

    def create_legend(
            self,
            axis: bool = False,
            rows: int = 1,
            cols: int = 0,
            loc: str | tuple[float, float] = 'outside upper center',
            label_permute: list[int] | slice = slice(None),
            **kwargs: Any) -> None:
        """
        Plots the legend

        Parameters
        ----------
        axis : bool, default = False
            Whether to plot the legend on the axes or the figure
        rows : int, default = 1
            Number of rows for the legend
        cols : int, default = 0
            Number of columns for the legend, if 0, rows will be used
        loc : str | tuple[float, float], default = 'outside upper center'
            Location to place the legend
        label_permute : list[int] | slice, default = slice(None)
            Permutation of the legend labels

        **kwargs
            Optional keyword arguments to pass to Figure.legend
        """
        fig_size: float = float(self.fig.get_size_inches()[0]) * self.fig.dpi
        label_handles: dict[str, list[Artist] | tuple[Artist, ...]]
        legend_range: ndarray
        handle: Artist
        artist: Figure | Axes

        if self.legend is not None:
            self.legend.remove()

        label_handles = self._label_handles()

        # Formatting of the legend
        if axis and self.legend_axis is None:
            match self.axes:
                case ndarray():
                    artist = self.axes.flatten()[0]
                case dict():
                    artist = list(self.axes.values())[0]
                case Axes():
                    artist = self.axes
                case _:
                    raise ValueError(f'Unknown axes attribute type ({type(self.axes)}), must be'
                                     f'either ndarray, dict, or Axes')
        elif axis:
            artist = self.legend_axis
        else:
            artist = self.fig

        # Create legend
        self.legend = artist.legend(
            np.array(list(label_handles.values()))[label_permute],
            np.array(list(label_handles.keys()))[label_permute],
            fancybox=False,
            ncol=cols or np.ceil(len(self._labels) / rows),
            fontsize=self._major,
            borderaxespad=0.2,
            loc=loc,
            handler_map=dict.fromkeys(
                [list, tuple, ndarray],
                utils.UniqueHandlerTuple(ndivide=None),
            ),
            **kwargs,
        )
        legend_range = np.array(self.legend.get_window_extent())[:, 0]

        # Recreate legend if it overflows the figure with more rows
        if legend_range[1] - legend_range[0] > fig_size:
            rows = np.ceil((legend_range[1] - legend_range[0]) * (
                np.ceil(len(self._labels) / cols) if cols else rows
            ) / fig_size)
            self.create_legend(axis=axis, rows=rows, loc=loc, label_permute=label_permute, **kwargs)

        # Update handles to remove transparency if there isn't any hatching and set point size
        for handle in self.legend.legend_handles:
            if not hasattr(handle, 'get_hatch'):
                handle.set_alpha(1)

            if isinstance(handle, PathCollection):
                handle.set_sizes([500])

            if isinstance(handle, Line2D):
                handle.set_linewidth(4)

    def set_axes_pad(self, pad: float = 0) -> None:
        """
        Sets the padding on all plot axes

        Parameters
        ----------
        pad : float, default = 0
            Fractional padding to the plot data
        """
        ranges: ndarray | None
        axis: Axes

        # Set the padding for each axis
        for axis in self.plots:
            ranges = self._axis_data_ranges(axis, pad=pad)

            if ranges is not None:
                axis.set_xlim(*ranges[:, 0])
                axis.set_ylim(*ranges[:, 1])

    def plot_density(
            self,
            colour: str,
            data: ndarray,
            ranges: ndarray,
            axis: Axes,
            label: str = '',
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
        label : str, default = ''
            Label for the data
        hatch : str, default = ''
            Hatching pattern for the contour
        order : list[int] | None, default = None
            Order of the axes, only required for 3D sciplots
        confidences : list[float] | None, default = [0.68]
            List of confidence values to plot contours for, starting with the lowest confidence

        **kwargs
            Optional kwargs to pass to Axes.contour and Axes.contourf
        """
        total: float
        levels: list[float]
        contour: ndarray
        grid: ndarray = np.mgrid[
                        ranges[0, 0]:ranges[0, 1]:self._bins * 1j,
                        ranges[1, 0]:ranges[1, 1]:self._bins * 1j,
                        ]
        kernel: gaussian_kde

        try:
            kernel = gaussian_kde(data.swapaxes(0, 1))
        except np.linalg.LinAlgError:
            self._logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.reshape(kernel(grid.reshape(2, -1)).T, (self._bins, self._bins))
        total = np.sum(contour)
        levels = [np.max(contour)]

        if confidences is None:
            confidences = [0.68]

        for confidence in confidences:
            levels.insert(0, utils.contour_sig(total * confidence, contour))

        if levels[-1] == 0:
            self._logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.concatenate((grid, contour[np.newaxis]), axis=0)

        if order is not None:
            contour = contour[order]

        self.plots[axis].append(axis.contourf(
            *contour,
            levels,
            alpha=self._alpha_2d,
            colors=colour,
            hatches=[hatch],
            **kwargs,
        ))
        self.plots[axis][-1]._hatch_color = colors.to_rgba(colour, self._alpha_line)
        self.plots[axis][-1].set_label(label)
        self.plots[axis].append(axis.contour(
            *contour,
            levels,
            colors=colour,
            linewidths=self._line_width,
            **kwargs,
        ))
        self.plots[axis][-1].set_label(label)

    def plot_errors(
            self,
            colour: str,
            x_data: ndarray,
            y_data: ndarray,
            axis: Axes,
            label: str = '',
            style: str = 'x',
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
        label : str, default = ''
            Label for the data
        style : str, default = 'x'
            Marker or line style for the data, can be either a marker style for scatter plots or
            line style for line plots
        x_error : (N) ndarray | (2,N) ndarray | None, default = None
            N x-errors, can be asymmetric
        y_error : (N) ndarray | (2,N) ndarray | None, default = None
            N y-errors, can be asymmetric

        **kwargs
            Optional keyword arguments to pass to Axes.scatter and Axes.errorbar
        """
        if 'marker' in kwargs:
            warn(
                'marker keyword argument is deprecated, please use style instead',
                DeprecationWarning,
                stacklevel=2,
            )
            style = kwargs.pop('marker')

        if style in utils.MARKERS:
            self.plots[axis].append(axis.scatter(
                x_data,
                y_data,
                label=label,
                color=colour,
                marker=style,
                s=self._marker_size,
                alpha=self._alpha_marker,
                **kwargs,
            ))
        elif style in utils.LINE_STYLES:
            self.plots[axis].append(axis.plot(
                x_data,
                y_data,
                lw=self._line_width,
                alpha=self._alpha_line,
                ls=style,
                label=label,
                color=colour,
                **kwargs,
            )[0])

        if self._error_region and y_error is not None:
            self.plots[axis].append(axis.fill_between(
                x_data,
                y_data - y_error if np.ndim(y_error) == 1 else y_error[0],
                y_data + y_error if np.ndim(y_error) == 1 else y_error[1],
                label=label,
                color=colour,
                alpha=self._alpha_2d
            ))
        elif x_error is not None or y_error is not None:
            self.plots[axis].append(axis.errorbar(
                x_data,
                y_data,
                yerr=y_error,
                xerr=x_error,
                capsize=self._cap_size,
                alpha=self._alpha_marker,
                capthick=self._eline_width,
                elinewidth=self._eline_width,
                label=label,
                color=colour,
                linestyle='',
                **kwargs,
            ))

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
        self.plots[axis].append(axis.imshow(
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
            label: str = '',
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
        label : str, default = ''
            Label for the data
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
        x_data: ndarray
        y_data: ndarray
        kernel: gaussian_kde
        axis.ticklabel_format(axis='y', scilimits=(-2, 2))

        if orientation not in {'vertical', 'horizontal'}:
            raise ValueError(f"Orientation ({orientation}) must be 'vertical' or 'horizontal'")

        if log:
            axis.set_xscale('log')

        if range_ is None and data.dtype.type not in {np.str_, np.object_}:
            range_ = (np.min(data), np.max(data))

        if self._density and len(np.unique(data)) > 1:
            kernel = gaussian_kde(np.log10(data) if log else data)
            x_data = np.linspace(*(np.log10(range_) if log else range_), self._bins)
            y_data = kernel(x_data)
            x_data = 10 ** x_data if log else x_data

            if norm:
                y_data /= np.max(y_data)

            if orientation == 'vertical':
                self.plots[axis].append(axis.fill_between(
                    x_data,
                    y_data,
                    lw=self._line_width,
                    hatch=hatch,
                    label=label,
                    facecolor=(colour, self._alpha_2d),
                    edgecolors=(colour, self._alpha_line),
                    **kwargs,
                ))
            else:
                self.plots[axis].append(axis.fill_betweenx(
                    x_data,
                    y_data,
                    lw=self._line_width,
                    hatch=hatch,
                    label=label,
                    facecolor=(colour, self._alpha_2d),
                    edgecolor=(colour, self._alpha_line),
                    **kwargs,
                ))
        elif norm:
            y_data, x_data = np.histogram(
                data,
                np.logspace(*np.log10(range_), self._bins) if log else self._bins,
            ) if data.dtype.type not in {np.str_, np.object_} else np.unique(
                data,
                return_counts=True,
            )[::-1]
            y_data = y_data / np.max(y_data)
            self.plots[axis].append((axis.bar if orientation == 'vertical' else axis.barh)(
                x_data[:-1] if x_data.dtype.type not in {np.str_, np.object_} else x_data,
                y_data,
                width=np.diff(x_data) if x_data.dtype.type not in {np.str_, np.object_} else 0.8,
                label=label,
                hatch=hatch,
                align='edge' if x_data.dtype.type not in {np.str_, np.object_} else 'center',
                color=colour,
                alpha=self._alpha_2d,
                **kwargs,
            ))
        else:
            self.plots[axis].append(axis.hist(
                data,
                bins=np.logspace(*np.log10(range_), self._bins) if log else self._bins,
                hatch=hatch,
                label=label,
                histtype='step',
                alpha=self._alpha_line,
                lw=self._line_width,
                orientation=orientation,
                facecolor=(colour, self._alpha_2d),
                edgecolor=(colour, self._alpha_line),
                range=range_,
                **kwargs,
            )[-1][0])
            self.plots[axis].append(axis.hist(
                data,
                bins=np.logspace(*np.log10(range_), self._bins) if log else self._bins,
                hatch=hatch,
                label=label,
                histtype='stepfilled',
                alpha=self._alpha_2d,
                lw=self._line_width,
                orientation=orientation,
                facecolor=(colour, self._alpha_2d),
                edgecolor=(colour, self._alpha_line),
                range=range_,
                **kwargs,
            )[-1][0])

    def plot_param_pairs(
            self,
            colour: str,
            data: ndarray,
            norm: bool = False,
            label: str = '',
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
        norm : bool, default = False
            If the histogram or density plot should be normalised to a maximum height of 1
        label : str, default = ''
            Label for the data
        hatch : str = ''
            Hatching of the histograms or density sciplots and contours
        ranges : (L,2) ndarray, default = None
            Ranges for L parameters, required if using kwargs to plot densities
        markers : (N) ndarray | None = None
            Markers for scatter sciplots for N data points

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
                # Share y-axis for all scatter sciplots
                if i != j:
                    axis.sharey(axes_row[0])

                # Set number of ticks
                axis.locator_params(axis='x', nbins=3)
                axis.locator_params(axis='y', nbins=3)

                # Hide ticks for sciplots that aren't in the first column or bottom row
                if j != 0 or j == i:
                    axis.tick_params(labelleft=False, left=False)

                if i != self.axes.shape[0] - 1:
                    axis.tick_params(labelbottom=False, bottom=False)

                if j < i:
                    axis.set_xlim(x_range)
                    axis.set_ylim(y_range)

                # Plot scatter sciplots & histograms
                if i == j:
                    self.plot_hist(
                        colour,
                        x_data,
                        axis,
                        norm=norm,
                        label=label,
                        hatch=hatch,
                        range_=x_range,
                        **kwargs,
                    )
                elif j < i:
                    for marker in np.unique(markers):
                        self.plots[axis].append(axis.scatter(
                            x_data[marker == markers][:self._scatter_num],
                            y_data[marker == markers][:self._scatter_num],
                            s=self._marker_size,
                            alpha=self._alpha_marker,
                            label=label,
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
                            label=label,
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
            label: str = '',
            target_colour: str | None = None,
            x_error: ndarray | None = None,
            y_error: ndarray | None = None,
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
        label : str, default = ''
            Label for the data
        target_colour : str | None, default = None
            Colour of the target data, if None, colour will be used
        x_error : (N) ndarray | (2,N) ndarray | None, default = None
            N x-errors, can be asymmetric
        y_error : (N) ndarray | (2,N) ndarray | None, default = None
            N y-errors, can be asymmetric
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

        if major_axis not in self.plots:
            self.plots[major_axis] = []

        self.plot_errors(
            colour,
            x_data,
            (pred - target) / target if error else pred - target,
            axis,
            x_error=x_error,
            y_error=y_error / target if error else y_error,
            **kwargs,
        )
        self.plot_errors(
            colour,
            x_data,
            pred,
            major_axis,
            label=label,
            x_error=x_error,
            y_error=y_error,
            **kwargs,
        )
        self.plots[major_axis].append(major_axis.plot(
            x_data,
            target,
            lw=self._line_width,
            color=target_colour or colour,
        )[0])

        axis.set_ylabel('Error' if error else 'Residual', fontsize=self._major)
        self.plots[axis].append(axis.hlines(
            0,
            xmin=x_data[0] if x_data.dtype.type in {np.str_, np.object_} else np.min(x_data),
            xmax=x_data[-1] if x_data.dtype.type in {np.str_, np.object_} else np.max(x_data),
            color='k',
        ))
        major_axis.tick_params(axis='y', labelsize=self._minor)
        major_axis.set_xticks([])
        return major_axis
