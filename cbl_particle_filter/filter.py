"""filter."""

from dataclasses import dataclass
from .carpet_map import CarpetMap

@dataclass
class Pose:
    x: float
    y: float
    heading: float

@dataclass
class OdomMeasurement:
    dx: float
    dy: float
    dheading: float

@dataclass
class ColorMeasurement:
    color_index: int

class CarpetBasedParticleFilter():
    def __init__(self, carpet_map:CarpetMap):
        self.carpet_map = carpet_map
        self.current_pose = Pose(x=0,y=0,heading=0)

    def update(self, odom:OdomMeasurement, color:ColorMeasurement):
        pass