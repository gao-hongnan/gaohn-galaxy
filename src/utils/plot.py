from typing import Any, Dict

import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.quiver import Quiver
from matplotlib_inline import backend_inline
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def make_meshgrid(low: float, high: float, num_samples: int) -> np.ndarray:
    """Make meshgrid for plotting contours and surfaces."""
    x = np.linspace(low, high, num_samples)
    y = np.linspace(low, high, num_samples)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def plot_quiver(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angles: str = "xy",
    scale_units: str = "xy",
    scale: float = 1,
    **kwargs: Dict[str, Any]
) -> Quiver:
    """Plot quiver."""
    return ax.quiver(
        x, y, u, v, angles=angles, scale_units=scale_units, scale=scale, **kwargs
    )


def plot_hist(ax: plt.Axes, x: np.ndarray, **kwargs: Dict[str, Any]) -> BarContainer:
    """Plot histogram."""
    return ax.hist(x, **kwargs)


def plot_bar(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs: Dict[str, Any]
) -> BarContainer:
    """Plot bar plot."""
    return ax.bar(x, y, **kwargs)


def plot_scatter(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs: Dict[str, Any]
) -> PathCollection:
    """Plot scatter plot."""
    return ax.scatter(x, y, **kwargs)


def plot_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    contourf: bool = False,
    **kwargs: Dict[str, Any]
) -> QuadContourSet:
    """Plot contours."""
    if contourf:
        contour = ax.contourf(x, y, z, **kwargs)
    else:
        contour = ax.contour(x, y, z, **kwargs)
    return contour


def plot_surface(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs: Dict[str, Any]
) -> mpl_toolkits.mplot3d.art3d.Poly3DCollection:
    """Plot surface."""
    surface = ax.plot_surface(x, y, z, **kwargs)
    return surface
