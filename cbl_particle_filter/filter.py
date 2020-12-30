"""filter."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.stats import norm, gamma, uniform, circmean
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


class CarpetBasedParticleFilter():
    def __init__(self, carpet_map: CarpetMap):
        self.carpet_map = carpet_map
        self.current_pose = Pose(x=0, y=0, heading=0)
        WEIGHT_FN_P = 1.0

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
                                       n_particles=200,
                                       dynamics_fn=odom_update,
                                       noise_fn=lambda x, odom: gaussian_noise(
                                           x, sigmas=[0.05, 0.05, 0.05]),
                                       weight_fn=weight_fn,
                                       resample_proportion=0.1,
                                       column_names=columns)

    def update(self, odom: OdomMeasurement, color: ColorMeasurement):
        self._pfilter.update(color.color_index,
                             odom=np.array([odom.dx, odom.dy, odom.dheading]))

    def get_current_pose(self):
        state = self._pfilter.mean_state
        state[2] = circmean(
            self._pfilter.particles[:, 2])  # take circular mean for heading
        return Pose(x=state[0], y=state[1], heading=state[2])

    def plot(self):
        self.carpet_map.plot()


def offline_playback(input_data: List[Tuple[OdomMeasurement, ColorMeasurement,
                                            Optional[Pose]]]):
    """
    Run the filter over given input data
    Input data provided as list of Tuples of odom, color, and optionally ground truth pose
    """
    plot = True

    if plot:
        from .visualisation import plot_map, plot_particles, plot_pose
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure()
        plt.ion()
        plt.show()
        frame_count = 0

    # todo: use input data and map from args
    from .simulator import make_input_data, make_map
    carpet = make_map()
    input_data = make_input_data(
        odom_pos_noise_std_dev=0,
        odom_heading_noise_std_dev=0,
        color_noise=0,
    )

    particle_filter = CarpetBasedParticleFilter(carpet)
    for odom, color, ground_truth_pose in input_data:
        particle_filter.update(odom, color)

        if plot:
            plot_map(carpet, show=False)
            plot_particles(particle_filter._pfilter.particles, show=False)
            estimated_pose = particle_filter.get_current_pose()
            estimated_pose_plt = plot_pose(
                estimated_pose.x,
                estimated_pose.y,
                estimated_pose.heading,
                color="red",
                show=False,
            )
            plot_pose(ground_truth_pose.x,
                      ground_truth_pose.y,
                      ground_truth_pose.heading,
                      show=False)
            plt.draw()
            plt.pause(0.01)
            plt.savefig(f"/tmp/filter_frame_{str(frame_count).zfill(6)}.png")
            frame_count += 1
            plt.cla()
