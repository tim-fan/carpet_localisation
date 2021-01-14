"""
carpet_map_csv_to_png

Script to read a .csv map as saved by excel and write a png file, which can later
be read in again as a carpet map.

This script exists because carpet mapping is performed in excel, and carpet maps are loaded
from png. This script performs the necessary conversion.

Usage:
    carpet_map_csv_to_png <input-csv> <output-png>
"""

from docopt import docopt
import numpy as np
from cbl_particle_filter.carpet_map import CarpetMap, save_map_as_png


def main():

    arguments = docopt(__doc__, version='1.0')
    csvfile = arguments['<input-csv>']
    pngfile = arguments['<output-png>']

    carpet_grid = np.loadtxt(csvfile, delimiter=',', dtype=int)

    # Note cell_size is not used in writing png, so we don't need to use the
    # actual value
    carpet_map = CarpetMap(grid=carpet_grid, cell_size=1)

    save_map_as_png(carpet_map, pngfile)

    print(f"Saved map to {pngfile}")