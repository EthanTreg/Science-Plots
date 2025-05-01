"""
Plots that represent the data as a grid of colours and values
"""
from typing import Any

import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr
from matplotlib.axes import Axes

from sciplots import utils
from sciplots.base import BasePlot
from sciplots.utils import label_change


class BaseGrid(BasePlot):
    """
    Base class to plot grids

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
    def create_legend(self, *_, **__) -> None:
        """
        Confusion matrix should not have a legend
        """
        return

    def set_axes_pad(self, _: float = 0) -> None:
        """
        Confusion matrix should not have axis padding
        """
        return


class PlotConfusion(BaseGrid):
    """
    Plots the confusion matrix between targets and predictions

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
            labels: list[str],
            preds: ndarray,
            targets: ndarray,
            fig_size: tuple[int, int] = utils.HI_RES_SQUARE,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        labels : list[str]
            Labels for each class
        preds : (N) ndarray
            Predicted value for N data points
        targets : (N) ndarray
            Target values for N data points
        fig_size : tuple[int, int], default = HI_RES_SQUARE
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._data: ndarray
        self._targets: ndarray = targets
        self._classes: ndarray = np.unique(self._targets)
        super().__init__(
            preds,
            labels=labels,
            fig_size=fig_size,
            **kwargs,
        )

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        class_: float
        idxs: ndarray
        counts: ndarray
        matrix_row: ndarray
        class_preds: ndarray
        matrix: ndarray = np.zeros((len(self._classes), len(self._classes)))

        # Generate confusion matrix
        for matrix_row, class_ in zip(matrix, self._classes):
            idxs = self._targets == class_
            class_preds, counts = np.unique(self._data[idxs], return_counts=True)
            class_preds = label_change(class_preds, self._classes)
            matrix_row[class_preds] = counts / np.count_nonzero(idxs) * 100

        self.plot_grid(
            matrix,
            self.axes,
            precision=1,
            x_labels=[
                '\n' + label if i % 2 == 1 else label for i, label in enumerate(self._labels)
            ],
            y_labels=[
                label + '\n' if i % 2 == 1 else label for i, label in enumerate(self._labels)
            ],
            range_=(0, 100),
        )
        self.axes.set_xlabel('Predictions', fontsize=self._major)
        self.axes.set_ylabel('Targets', fontsize=self._major)


class PlotPearson(BaseGrid):
    """
    Plots the Pearson correlation coefficient between two sets of data.

    Attributes
    ----------
    plots : dict[Axes, list[Artist | Container]], default = {}
        Plot artists for each axis
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default=None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default=None
        Plot legend
    """
    def __init__(
            self,
            x_data: ndarray,
            y_data: ndarray,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        x_data : (N, P) ndarray
            N data points for P parameters for the x-axis
        y_data : (N, Q) ndarray
            N data points for Q parameters for the y-axis
        x_labels : list[str] | None, default=None
            Labels for the x-axis parameters
        y_labels : list[str] | None, default=None
            Labels for the y-axis parameters
        fig_size : tuple[int, int], default=utils.HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._x_labels: list[str] = x_labels or ['']
        self._y_labels: list[str] = y_labels or ['']
        self._data: ndarray = x_data.swapaxes(0, 1)
        self._y_data: ndarray = y_data.swapaxes(0, 1)
        super().__init__(self._data, fig_size=fig_size, **kwargs)

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        matrix: ndarray = np.empty((len(self._y_data), len(self._data)))

        for i, y_data in enumerate(self._y_data):
            for j, x_data in enumerate(self._data):
                matrix[i, j] = pearsonr(x_data, y_data)[0]

        self.plot_grid(
            matrix,
            self.axes,
            diverge=True,
            x_labels=self._x_labels,
            y_labels=self._y_labels,
            range_=(-1, 1),
        )
