# define a mapping of color enum values to RGB values, for visualisation/debug
# purposes.
# Will hardcode this mapping for now, if this is ever migrated to different carpets
# this will need to be made configurable

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Color:
    index: int
    name: str
    rgb: Tuple[int, int, int]


BLACK = Color(0, "BLACK", (80, 80, 80))
LIGHT_BLUE = Color(1, "LIGHT_BLUE", (51, 204, 255))
BEIGE = Color(2, "BEIGE", (241, 230, 218))
DARK_BLUE = Color(3, "DARK_BLUE", (0, 51, 204))
UNCLASSIFIED = Color(4, "UNCLASSIFIED", (0, 0, 0))

COLORS = [BLACK, LIGHT_BLUE, BEIGE, DARK_BLUE]

color_from_index = {color.index: color for color in COLORS + [UNCLASSIFIED]}

color_from_name = {color.name: color for color in COLORS + [UNCLASSIFIED]}

color_from_rgb = {color.rgb: color for color in COLORS + [UNCLASSIFIED]}
