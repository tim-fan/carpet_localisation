"""filter."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, gamma, uniform
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample 
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


        # initialisation of particle filter implementation
        # refer https://github.com/johnhw/pfilter/blob/master/README.md

        columns = ["x", "y", "heading"]
        map_x_size = self.carpet_map.grid.shape[1] * self.carpet_map.cell_size
        map_y_size = self.carpet_map.grid.shape[0] * self.carpet_map.cell_size
        prior_fn = independent_sample([
            uniform(loc=0, scale=map_x_size).rvs, 
            uniform(loc=0, scale=map_y_size).rvs,
            uniform(loc=0, scale=2*np.pi).rvs, 
        ])

        def observe_fn(state: np.array) -> np.array:
            return self.carpet_map.get_color_at_coords(state)

        def weight_fn(hyp_observed:np.array, real_observed:np.array):
            """
            weight p for correct observations, 1-p for incorrect
            """
            p = 0.9

            correct_observation = hyp_observed == real_observed

            return correct_observation * p + ~correct_observation * (1-p)

        def no_motion_update(x):
            """
            Placeholder until odom-based updates are implemented
            """
            return x

        self._pfilter = ParticleFilter(
            prior_fn=prior_fn, 
            observe_fn=observe_fn,
            n_particles=200,
            dynamics_fn=no_motion_update,
            noise_fn=lambda x: 
                        gaussian_noise(x, sigmas=[0.2, 0.2, 0.05]),
            weight_fn=weight_fn,
            resample_proportion=0.1,
            column_names = columns
        )

    def update(self, odom:OdomMeasurement, color:ColorMeasurement):
        self._pfilter.update()
