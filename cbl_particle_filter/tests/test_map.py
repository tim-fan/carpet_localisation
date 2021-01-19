import importlib.resources as pkg_resources
import tempfile
import os
import numpy as np
from ..carpet_map import CarpetMap, generate_random_map, save_map_as_png, load_map_from_png
from . import data


def make_test_map() -> CarpetMap:
    return CarpetMap(
        grid=np.array([
            [1, 9, 2],
            [8, 3, 7],
            [4, 6, 5],
        ]),
        cell_size=0.4,
    )


def test_init():
    carpet = make_test_map()
    assert isinstance(carpet, CarpetMap)


def test_get_color_at_coords():
    carpet = make_test_map()

    coords = np.array([
        # two points in the bottom left cell, near the origin:
        [0.0, 0.0],
        [0.3, 0.1],
        # a series of randomly chosen points:
        [0.5, 0.7],
        [0.3, 1.1],
        [0.9, 0.01],
        [0.8001, 0.80001],
        [0.5, 1.0],
        # following four coords test out of bounds points - should
        # return color = -1
        [-1, 0.1],
        [0.1, -4],
        [1.21, 0.1],
        [0.1, 1.21],
    ])

    expected_colors = np.array([
        4,
        4,
        3,
        1,
        5,
        2,
        9,
        -1,
        -1,
        -1,
        -1,
    ])

    np.testing.assert_array_equal(expected_colors,
                                  carpet.get_color_at_coords(coords))


def test_generate_random_map():
    shape = (10, 20)
    cell_size = 0.5
    n_colors = 4  # based on hardcoded colors in carpet_map.py
    carpet = generate_random_map(shape, cell_size)

    assert isinstance(carpet, CarpetMap)
    assert carpet.cell_size == cell_size
    assert carpet.grid.shape == shape
    assert len(np.unique(carpet.grid)) == n_colors

    # for debug visualisation set plot=True
    plot = False
    if plot:
        from ..visualisation import plot_map
        plot_map(carpet)


def test_save_map_as_png():
    shape = (20, 40)
    cell_size = 0.5
    np.random.seed(123)
    carpet = generate_random_map(shape, cell_size)

    with tempfile.TemporaryDirectory() as tmpdirname:
        outfile = f"{tmpdirname}/saved_map.png"

        # To save the generated map for viewing, can
        # override outfile here:
        # outfile = '/tmp/saved_map.png'

        save_map_as_png(carpet, filepath=outfile)
        assert os.path.isfile(outfile)


def test_load_map_from_png():

    with pkg_resources.path(data, "random_map.png") as map_png_path:

        carpet = load_map_from_png(str(map_png_path), cell_size=0.5)

    # generate the exepected result
    # (the saved carpet was generated the same way)
    shape = (20, 40)
    cell_size = 0.5
    np.random.seed(123)
    expected_carpet = generate_random_map(shape, cell_size)

    assert carpet == expected_carpet
