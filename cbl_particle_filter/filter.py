"""filter."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pickle
from scipy.stats import norm, gamma, uniform, circmean
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from .carpet_map import CarpetMap
from .colors import color_from_index


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


def add_poses(current_poses: np.array, pose_increments: np.array) -> np.array:
    """
    add pose_increments to current_poses, interpreting pose_increments in the frame of
    current_poses

    poses expected as nx3 matrix for n poses, with colums [x,y,heading]
    """
    current_x = current_poses[:, 0]
    current_y = current_poses[:, 1]
    current_heading = current_poses[:, 2]

    inc_x = pose_increments[:, 0]
    inc_y = pose_increments[:, 1]
    inc_heading = pose_increments[:, 2]

    result_x = current_x + inc_x * np.cos(current_heading) - inc_y * np.sin(
        current_heading)
    result_y = current_y + inc_x * np.sin(current_heading) + inc_y * np.cos(
        current_heading)
    result_heading = current_heading + inc_heading

    return np.column_stack([result_x, result_y, result_heading])


def load_input_log(
        log_path: str) -> List[Tuple[OdomMeasurement, ColorMeasurement, Pose]]:
    """
    Load an input log, as recorded by a CarpetBasedParticleFilter
    """
    with open(log_path, "rb") as f:
        return pickle.load(f)


class CarpetBasedParticleFilter():
    def __init__(self, carpet_map: CarpetMap, log_inputs: bool = False):
        """
        Initialise with a given map.
        if `log_inputs` is set True, all inputs to `update` will be logged,
        and can be saved with CarpetBasedParticleFilter.write_input_log(logpath)
        """
        self.carpet_map = carpet_map
        self.log_inputs = log_inputs
        self.input_log = []

        WEIGHT_FN_P = 0.95
        N_PARTICLES = 500

        # initialisation of particle filter implementation
        # refer https://github.com/johnhw/pfilter/blob/master/README.md

        columns = ["x", "y", "heading"]
        map_x_size = self.carpet_map.grid.shape[1] * self.carpet_map.cell_size
        map_y_size = self.carpet_map.grid.shape[0] * self.carpet_map.cell_size
        prior_fn = independent_sample([
            uniform(loc=0, scale=map_x_size).rvs,
            uniform(loc=0, scale=map_y_size).rvs,
            uniform(loc=0, scale=2 * np.pi).rvs,
        ])

        def observe_fn(state: np.array, **kwargs) -> np.array:
            return self.carpet_map.get_color_at_coords(state)

        def weight_fn(hyp_observed: np.array, real_observed: np.array,
                      **kwargs):
            """
            weight p for correct observations, 1-p for incorrect
            """
            p = WEIGHT_FN_P

            correct_observation = np.squeeze(hyp_observed) == np.squeeze(
                real_observed)

            weights = correct_observation * p + ~correct_observation * (1 - p)
            return weights

        def odom_update(x: np.array, odom: np.array) -> np.array:
            return add_poses(x, np.array([odom]))

        self._pfilter = ParticleFilter(prior_fn=prior_fn,
                                       observe_fn=observe_fn,
                                       n_particles=N_PARTICLES,
                                       dynamics_fn=odom_update,
                                       noise_fn=lambda x, odom: gaussian_noise(
                                           x, sigmas=[0.05, 0.05, 0.05]),
                                       weight_fn=weight_fn,
                                       resample_proportion=0.1,
                                       column_names=columns)

    def update(self,
               odom: OdomMeasurement,
               color: ColorMeasurement,
               ground_truth: Optional[Pose] = None) -> None:
        """
        Update the particle filter based on measured odom and color
        If optional ground truth pose is provided, and if input logging is enabled, the 
        ground truth pose will be logged.
        """
        self._pfilter.update(color.color_index,
                             odom=np.array([odom.dx, odom.dy, odom.dheading]))

        if self.log_inputs:
            self.input_log.append((odom, color, ground_truth))

    def get_current_pose(self) -> Pose:
        mean_x = np.mean(self._pfilter.particles[:, 0])
        mean_y = np.mean(self._pfilter.particles[:, 1])
        mean_heading = circmean(
            self._pfilter.particles[:, 2])  # take circular mean for heading
        return Pose(x=mean_x, y=mean_y, heading=mean_heading)

    def get_particles(self) -> np.ndarray:
        return self._pfilter.particles

    def write_input_log(self, log_path: str) -> None:
        assert self.log_inputs, "Error - requested 'write_input_log', but input logging is disabled"
        with open(log_path, 'wb') as f:
            pickle.dump(self.input_log, f, protocol=pickle.HIGHEST_PROTOCOL)


def offline_playback(input_data: List[Tuple[OdomMeasurement, ColorMeasurement,
                                            Optional[Pose]]],
                     carpet: CarpetMap,
                     plot: bool = True):
    """
    Run the filter over given input data
    Input data provided as list of Tuples of odom, color, and optionally ground truth pose
    """

    if plot:
        from .visualisation import plot_map, plot_particles, plot_pose
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure()
        plt.ion()
        plt.show()
        frame_count = 0

    particle_filter = CarpetBasedParticleFilter(carpet)
    for odom, color, ground_truth_pose in input_data:
        print(
            f"update with color: {color_from_index(color.color_index).name}, odom:{odom}"
        )

        particle_filter.update(odom, color)

        if plot:
            plot_map(carpet, show=False)
            plot_particles(particle_filter._pfilter.particles, show=False)
            estimated_pose = particle_filter.get_current_pose()
            estimated_pose_plt = plot_pose(
                estimated_pose,
                color="red",
                show=False,
            )
            if ground_truth_pose:
                plot_pose(ground_truth_pose, show=False)
            plt.draw()
            plt.pause(0.01)
            plt.savefig(f"/tmp/filter_frame_{str(frame_count).zfill(6)}.png")
            frame_count += 1
            plt.cla()
