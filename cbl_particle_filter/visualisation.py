"""visualisation
plotting functions for the particle filter
"""
from .carpet_map import CarpetMap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_map(carpet_map: CarpetMap, show=True):
    colormap = ListedColormap(
        np.array([
            [80, 80, 80, 255],
            [51, 204, 255, 255],
            [241, 230, 218, 255],
            [0, 51, 204, 255],
        ]) / 255)
    map_x_size = carpet_map.grid.shape[1] * carpet_map.cell_size
    map_y_size = carpet_map.grid.shape[0] * carpet_map.cell_size
    plt.imshow(
        np.flipud(carpet_map.grid),
        origin="lower",
        extent=[0, map_x_size, 0, map_y_size],
        cmap=colormap,
    )
    if show:
        plt.show()


def plot_particles(state: np.array, show=True, color='red'):
    """
    plots particles as given by particle filter state matrix
    """
    arrow_length = 0.2
    x = state[:, 0]
    y = state[:, 1]
    heading = state[:, 2]
    dx = np.cos(heading) * arrow_length
    dy = np.sin(heading) * arrow_length
    plt.quiver(x, y, dx, dy, color=color)
    if show:
        plt.show()


def plot_pose(x, y, heading, show=True, color="green"):
    """
    plot a single pose as an arrow
    for multiple poses, use 'plot_particles'
    """
    arrow_length = 0.2
    width = 0.05
    dx = np.cos(heading) * arrow_length
    dy = np.sin(heading) * arrow_length
    plt.arrow(x,
              y,
              dx,
              dy,
              width=width,
              facecolor=color,
              linewidth=1,
              edgecolor="black")
    if show:
        plt.show()