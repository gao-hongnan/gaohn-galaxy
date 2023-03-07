"""Plots Graphical representation of the decision boundary of a classifer in 2D.

Code adapted from https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py

The purpose of the re-implementation is to understand what's going on under the hood.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.base.estimator import BaseEstimator
from src.base.types import T
from src.utils.plot import plot_contour, plot_scatter


def make_meshgrid(x1: T, x2: T, step: float = 0.02) -> Tuple[T, T]:
    """Create a mesh of points to plot in

    Parameters
    ----------
    x1 : T
        data to base x1-axis meshgrid on
    x2 : T
        data to base x2-axis meshgrid on
    step : float, optional (default=0.02)
        stepsize for meshgrid, by default 0.02

    Returns
    -------
    xx1, xx2 : T
        generated meshgrid
    """
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step)
    )
    return xx1, xx2


# pylint: disable=too-many-arguments
def plot_decision_regions(
    X: T,
    y: T,
    classifier: BaseEstimator,
    # test_idx: Optional[int] = None,
    markers: Optional[Tuple[str, ...]] = None,
    colors: Optional[Tuple[str, ...]] = None,
    cmap: Optional[ListedColormap] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """Plot decision regions."""
    ax = ax or plt.gca()

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v") if markers is None else markers
    colors = ("red", "blue", "lightgreen", "gray", "cyan") if colors is None else colors
    cmap = ListedColormap(colors[: len(np.unique(y))]) if cmap is None else cmap

    xx1, xx2 = make_meshgrid(X[:, 0], X[:, 1])

    X_input_space = np.array([xx1.ravel(), xx2.ravel()]).T  # N x 2 matrix

    Z = classifier.predict(X_input_space)
    Z = Z.reshape(xx1.shape)  # reshape to match xx1 and xx2 to plot contour

    contour = plot_contour(ax, xx1, xx2, Z, cmap=cmap, **kwargs)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.clabel(contour, inline=True, fontsize=8)

    for idx, cl in enumerate(np.unique(y)):
        plot_scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            ax=ax,
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )
    plt.show()
