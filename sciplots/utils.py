"""
Utility functions for creating sciplots
"""
from logging import getLogger
from typing import Sequence, TypeVar, Any, Callable

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.patches import Ellipse
from matplotlib.transforms import Transform
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import minimize

MARKERS: list[str] = list(Line2D.markers.keys())
HATCHES: list[str] = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '/o', '\\|', '|*', '-\\',
                      '+o', 'x*', 'o-', 'O|', 'O.', '*-']
RECTANGLE: tuple[int, int] = (16, 9)
SQUARE: tuple[int, int] = (10, 10)
HI_RES: tuple[int, int] = (32, 18)
HI_RES_SQUARE: tuple[int, int] = (20, 20)

IterableLike = TypeVar('IterableLike', list[object], dict[Any, object], ndarray)


class UniqueHandlerTuple(HandlerTuple):
    """
    Handler for legend to create handles with unique types per label
    """
    @staticmethod
    def _unique_handles(handles: Sequence[Artist]) -> Sequence[Artist]:
        """
        Returns a list of handles with unique types

        Parameters
        ----------
        handles : Sequence[Artist]
            List of handles

        Returns
        -------
        Sequence[Artist]
            Handles of unique types
        """
        handle: Artist
        return np.array(handles)[np.unique(
            [str(type(handle)) for handle in handles],
            return_index=True,
        )[1]].tolist()

    def create_artists(  # pylint: disable=missing-function-docstring
            self,
            legend: Legend,
            orig_handle: Sequence[Artist],
            xdescent: int,
            ydescent: int,
            width: int,
            height: int,
            fontsize: int,
            trans: Transform) -> Sequence[Artist]:
        artist_list: Sequence[Artist]

        # Generate legend handles and filter unique handles from original handles
        artist_list = self._unique_handles(super().create_artists(
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ))

        # Original handles may be unique, but their legend handles may not, so recreate legend
        # handles with unique handles
        if len(artist_list) != len(orig_handle):
            artist_list = super().create_artists(
                legend,
                artist_list,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            )
        return artist_list


def cast_func(
        func: str | Callable,
        objs: IterableLike,
        obj_first: bool = True,
        kwargs_unique: bool = False,
        args: list[list[Any]] | list[Any] | None = None,
        kwargs: list[dict[str, Any]] | dict[str, list[Any] | Any] | None = None) -> IterableLike:
    """
    Casts args and kwargs to a given function for a set of objects

    Parameters
    ----------
    func : str | Callable
        Function to call for the objects, if string, function will be treated as a method for the
        object
    objs : IterableLike
        List of objects to apply the function
    obj_first : bool, default = True
        If the args argument is a list where the indices correspond to the object index, or if
        the indices correspond to the argument index
    kwargs_unique : bool, default = False
        If the kwargs argument is a dictionary of lists with values for each object, or if each
        kwarg has the same value for all objects, only used if kwargs is a dictionary
    args : list[list[Any]] | list[Any] | None, default = None
        Arguments to pass to the function for each object, can either have indices corresponding
        to the object index, or corresponding to the argument index
    kwargs : list[dict[str, Any]] | dict[str, list[Any] | Any] | None, default = None
        Optional keyword arguments to pass to the function, can either be a list of dictionaries
        with indices corresponding to the object index, or a dictionary of lists with values for
        each object, or a dictionary of values used for all objects

    Returns
    -------
    IterableLike
        List of return values from the function call for each object
    """
    num: int = objs.size if isinstance(objs, ndarray) else len(objs)
    func_name: str
    key: str | Any
    arg: list[Any]
    idxs: tuple[int, ...]
    kwarg: dict[str, Any]
    obj: object
    value: Any
    returns: IterableLike

    if isinstance(func, str):
        func_name = func

        def func(x, *y, **z):
            return getattr(x, func_name)(*y, **z)

    if args is None:
        args = [()] * num
    elif obj_first or not all(isinstance(arg, list) for arg in args):
        args = list(zip(args))

    if len(args) == 1:
        args *= num

    if kwargs is None:
        kwargs = [{}] * num
    elif isinstance(kwargs, dict):
        if kwargs_unique or not all(isinstance(arg, list) for arg in kwargs.values()):
            for key, value in kwargs.items():
                kwargs[key] = [value]

        if len(set(len(arg) for arg in kwargs.values())) > 1:
            getLogger(__name__).warning(f'Not all lists in kwargs ({kwargs}) have the same length, '
                                        f'only elements up to the length of the shortest length '
                                        f'will be used')
        kwargs = [dict(zip(kwargs.keys(), arg)) for arg in zip(*kwargs.values())]

    if len(kwargs) == 1:
        kwargs *= num

    match objs:
        case ndarray():
            returns = np.empty_like(objs)

            for idxs, arg, kwarg in zip(np.ndindex(objs.shape), args, kwargs):
                returns[*idxs] = func(objs[*idxs], *arg, **kwarg)
        case dict():
            returns = {}

            for (key, obj), arg, kwarg in zip(objs.items(), args, kwargs):
                returns[key] = func(obj, *arg, **kwarg)
        case list():
            returns = []

            for obj, arg, kwarg in zip(objs, args, kwargs):
                returns.append(func(obj, *arg, **kwarg))
        case _:
            raise ValueError(f'Unknown type for objs ({type(objs)})')

    return returns


def contour_sig(counts: float, contour: ndarray) -> float:
    """
    Finds the level that includes the required counts in a contour

    Parameters
    ----------
    counts : float
        Target amount for level to include
    contour : ndarray
        Contour to find the level that gives the target counts

    Returns
    -------
    float
        Level
    """
    return minimize(
        lambda x: np.abs(np.sum(contour[contour > x]) / counts - 1),
        0,
        method='nelder-mead',
    )['x'][0]


def label_change(
        data: ndarray,
        in_label: ndarray,
        one_hot: bool = False,
        out_label: ndarray | None = None) -> ndarray:
    """
    Converts an array of class values to an array of class indices

    Parameters
    ----------
    data : (N) ndarray
        Classes of size N
    in_label : (C) ndarray
        Unique class values of size C found in data
    one_hot : bool, default = False
        If the returned tensor should be 1D array of class indices or 2D one hot tensor if out_label
        is None or is an int
    out_label : (C) ndarray, default = None
        Unique class values of size C to transform data into, if None, then values will be indexes

    Returns
    -------
    (N) | (N,C) ndarray
        ndarray of class indices, or if one_hot is True, one hot array
    """
    data_one_hot: ndarray
    out_data: ndarray

    if out_label is None:
        out_label = np.arange(len(in_label))

    assert out_label is not None
    out_data = out_label[np.searchsorted(in_label, data)]

    if one_hot:
        data_one_hot = np.zeros((len(data), len(in_label)))
        data_one_hot[np.arange(len(data)), out_data] = 1
        out_data = data_one_hot

    return out_data


def plot_ellipse(
        colour: str,
        data: ndarray,
        axis: Axes,
        stds: list[int] | None = None) -> None:
    """
    Creates confidence ellipse

    Parameters
    ----------
    colour : str
        Colour of the confidence ellipse border
    data : (N,2) ndarray
        N (x,y) data points to generate confidence ellipse for
    axis : Axes
        Axis to add confidence ellipse
    stds : list[int], default = [1]
        The standard deviations of the confidence ellipses
    """
    cov: ndarray
    eig_val: ndarray
    eig_vec: ndarray

    data = data.swapaxes(0, 1)
    cov = np.cov(*data)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_val = np.sqrt(eig_val)

    if stds is None:
        stds = [1]

    for std in stds:
        axis.add_artist(Ellipse(
            np.mean(data, axis=1),
            width=eig_val[0] * std * 2,
            height=eig_val[1] * std * 2,
            angle=np.rad2deg(np.arctan2(*eig_vec[::-1, 0])),
            facecolor='none',
            edgecolor=colour,
        ))


def subplot_grid(num: int) -> ndarray:
    """
    Calculates the most square grid for a given input for mosaic subplots

    Parameters
    ----------
    num : integer
        Total number to split into a mosaic grid

    Returns
    -------
    ndarray
        2D array of indices with relative width for mosaic subplot
    """
    # Constants
    grid = (int(np.sqrt(num)), int(np.ceil(np.sqrt(num))))
    subplot_layout = np.arange(num)
    diff_row = np.abs(num - np.prod(grid))

    # If number is not divisible into a square-ish grid,
    # then the total number will be unevenly divided across the rows
    if diff_row and diff_row != grid[0]:
        shift_num = diff_row * (grid[1] + np.sign(num - np.prod(grid)))

        # Layout of index and repeated values to correspond to the width of the index
        subplot_layout = np.vstack((
            np.repeat(
                subplot_layout[:-shift_num],
                int(shift_num / diff_row)
            ).reshape(grid[0] - diff_row, -1),
            np.repeat(
                subplot_layout[-shift_num:],
                int((num - shift_num) / (grid[0] - diff_row))
            ).reshape(diff_row, -1),
        ))
    # If a close to square grid is found
    elif diff_row:
        subplot_layout = subplot_layout.reshape(grid[0], grid[1] + 1)
    # If grid is square
    else:
        subplot_layout = subplot_layout.reshape(*grid)

    return subplot_layout
