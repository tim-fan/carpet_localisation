import numpy as np
from scipy.stats import uniform

from ..simulator import make_map
from ..visualisation import plot_map, plot_particles, plot_pose

import pytest

@pytest.fixture()
def show(pytestconfig):
    return pytestconfig.getoption("show_plot")

def make_random_particles() -> np.array:
    np.random.seed(123)

    num_particles = 100
    x = uniform.rvs(0, 2.5, num_particles)
    y = uniform.rvs(0, 2, num_particles)
    heading = uniform.rvs(0, np.pi * 2, num_particles)
    state = np.column_stack([x, y, heading])
    return state


def test_plot_map(show):
    test_map = make_map()
    plot_map(test_map, show=show)

def test_plot_particles(show):
    state = make_random_particles()
    plot_particles(state, show=show)


def test_plot_particles_on_map(show):
    test_map = make_map()
    state = make_random_particles()
    plot_map(test_map, show=False)
    plot_particles(state, show=show)


def test_plot_pose_on_map(show):
    test_map = make_map()
    plot_map(test_map, show=False)
    plot_pose(1, 1.5, np.pi, show=show)

