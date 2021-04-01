"""filter."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pickle
from scipy.stats import norm, gamma, uniform
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from .carpet_map import CarpetMap
from .colors import color_from_index, Color, UNCLASSIFIED


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


def load_input_log(log_path: str) -> List[Tuple[OdomMeasurement, Color, Pose]]:
    """
    Load an input log, as recorded by a CarpetBasedParticleFilter
    """
    with open(log_path, "rb") as f:
        return pickle.load(f)


class CarpetBasedParticleFilter():
    def __init__(self,
                 carpet_map: CarpetMap,
                 log_inputs: bool = False,
                 resample_proportion: float = 0,
                 weight_fn_p: float = 0.95,
                 odom_pos_noise: float = 0.05,
                 odom_heading_noise: float = 0.05,
                 n_particles: int = 500):
        """
        Initialise with a given map.
        if `log_inputs` is set True, all inputs to `update` will be logged,
        and can be saved with CarpetBasedParticleFilter.write_input_log(logpath)
        """
        self.carpet_map = carpet_map
        self.log_inputs = log_inputs
        self.input_log = []
        self._pfilter = None
        self._resample_proportion = resample_proportion
        self._weight_fn_p = weight_fn_p
        self._odom_pos_noise = odom_pos_noise
        self._odom_heading_noise = odom_heading_noise
        self._n_particles = n_particles
        self._most_recent_color = None  # used in pfilter initialisation fn

    def _pfilter_init(self):
        """
        Setup the particle filter.

        """

        # initialisation of particle filter implementation
        # refer https://github.com/johnhw/pfilter/blob/master/README.md

        columns = ["x", "y", "heading", "life_span"]
        map_x_size = self.carpet_map.grid.shape[1] * self.carpet_map.cell_size
        map_y_size = self.carpet_map.grid.shape[0] * self.carpet_map.cell_size

        def prior_fn(n_particles: int):
            """
            Sample n random particles from p(x|z)
            i.e. if last color is BEIGE, return a sample
            where proportion self._weight_fn_p of particles lie on
            random beige cells, and proportion (1-self._weight_fn_p)
            lie on other cells
            """
            # create a grid of sample probablilities, equal in shape
            # to the map grid, such that the sum of all cells
            # which match the most recent color equals self._weight_fn_p
            # and the sum of all cells = 1.
            p_mat = np.zeros_like(self.carpet_map.grid, dtype=float)

            if self._most_recent_color is None:
                matching = np.zeros_like(self.carpet_map.grid)
            else:
                matching = self.carpet_map.grid == self._most_recent_color.index

            num_matches = np.sum(matching)
            if num_matches == 0 or num_matches == self.carpet_map.grid.size:
                p_mat[:] = 1 / self.carpet_map.grid.size
            else:
                p_mat[matching] = self._weight_fn_p / np.sum(matching)
                p_mat[~matching] = (1 - self._weight_fn_p) / np.sum(~matching)

            # sample from the grid using the probabilities from above
            p_mat_flat = p_mat.flatten()
            selected_grid_linear_indices = np.random.choice(range(
                len(p_mat_flat)),
                                                            size=n_particles,
                                                            p=p_mat_flat)
            #convert linear indices back to grid indices
            y_indices, x_indices = np.unravel_index(
                selected_grid_linear_indices, self.carpet_map.grid.shape)

            # convert sampled grid indices into x/y coordinates
            # add noise to sample uniformly across selected grid cells
            x_coords = (x_indices +
                        uniform().rvs(n_particles)) * self.carpet_map.cell_size
            y_coords = (self.carpet_map.grid.shape[0] -
                        (y_indices + uniform().rvs(n_particles))
                        ) * self.carpet_map.cell_size

            heading = uniform(loc=0, scale=2 * np.pi).rvs(n_particles)
            age = np.zeros_like(heading, dtype=float)

            return np.column_stack([x_coords, y_coords, heading, age])

        def observe_fn(state: np.array, **kwargs) -> np.array:
            return self.carpet_map.get_color_at_coords(state[:, 0:3])

        def weight_fn(hyp_observed: np.array, real_observed: np.array,
                      **kwargs):
            """
            weight p for correct observations, 1-p for incorrect
            """
            p = self._weight_fn_p

            correct_observation = np.squeeze(hyp_observed) == np.squeeze(
                real_observed)

            weights = correct_observation * p + ~correct_observation * (1 - p)
            return weights

        def odom_update(x: np.array, odom: np.array) -> np.array:
            poses = add_poses(x[:, 0:3], np.array([odom]))
            life_span = x[:, 3]
            life_span += 1
            return np.column_stack((poses, life_span))

        self._pfilter = ParticleFilter(
            prior_fn=prior_fn,
            observe_fn=observe_fn,
            n_particles=self._n_particles,
            dynamics_fn=odom_update,
            noise_fn=lambda x, odom: gaussian_noise(
                x,
                sigmas=[
                    self._odom_pos_noise, self._odom_pos_noise, self.
                    _odom_heading_noise, 0
                ]),
            weight_fn=weight_fn,
            resample_proportion=self._resample_proportion,
            column_names=columns)

    def update(self,
               odom: OdomMeasurement,
               color: Color,
               ground_truth: Optional[Pose] = None) -> None:
        """
        Update the particle filter based on measured odom and color
        If optional ground truth pose is provided, and if input logging is enabled, the 
        ground truth pose will be logged.
        """
        if self.log_inputs:
            self.input_log.append((odom, color, ground_truth))

        self._most_recent_color = color

        # if particle filter is not intialised, init it
        if self._pfilter is None:
            self._pfilter_init()

        odom_array = [odom.dx, odom.dy, odom.dheading]
        if color == UNCLASSIFIED:
            self._pfilter.update(observed=None, odom=odom_array)
        else:
            self._pfilter.update(np.array([color.index]), odom=odom_array)

    def seed(self,
             seed_pose: Pose,
             pos_std_dev: float = 1.,
             heading_std_dev: float = 0.3) -> None:
        """
        Reinitialise particles around given seed pose
        Particles are initialised with standard deviations around
        position and heading as specified
        """
        # if particle filter is not intialised, init it
        if self._pfilter is None:
            self._pfilter_init()
        self._pfilter.particles = independent_sample([
            norm(loc=seed_pose.x, scale=pos_std_dev).rvs,
            norm(loc=seed_pose.y, scale=pos_std_dev).rvs,
            norm(loc=seed_pose.heading, scale=heading_std_dev).rvs,
            norm(loc=0, scale=0).rvs,
        ])(self._n_particles)

    def get_current_pose(self) -> Pose:
        if self._pfilter is None:
            return None
        oldest_particle = np.argmax(self._pfilter.particles[:, 3])
        state = self._pfilter.particles[oldest_particle, :]

        state = self._pfilter.mean_state
        return Pose(x=state[0], y=state[1], heading=state[2])

    def get_particles(self) -> np.ndarray:
        if self._pfilter is None:
            return None
        return self._pfilter.particles

    def write_input_log(self, log_path: str) -> None:
        assert self.log_inputs, "Error - requested 'write_input_log', but input logging is disabled"
        with open(log_path, 'wb') as f:
            pickle.dump(self.input_log, f, protocol=pickle.HIGHEST_PROTOCOL)


def offline_playback(input_data: List[Tuple[OdomMeasurement, Color,
                                            Optional[Pose]]],
                     carpet: CarpetMap,
                     seed_pose: Optional[Pose] = None,
                     resample_proportion: float = 0,
                     weight_fn_p: float = 0.95,
                     odom_pos_noise: float = 0.05,
                     odom_heading_noise: float = 0.05,
                     n_particles: int = 500,
                     plot: bool = False,
                     verbose=False) -> List[Pose]:
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

    particle_filter = CarpetBasedParticleFilter(
        carpet,
        resample_proportion=resample_proportion,
        weight_fn_p=weight_fn_p,
        odom_pos_noise=odom_pos_noise,
        odom_heading_noise=odom_heading_noise,
        n_particles=n_particles)

    if seed_pose:
        particle_filter.seed(seed_pose)

    post_update_poses = []

    for odom, color, ground_truth_pose in input_data:
        if verbose:
            print(f"update with color: {color.name}, odom:{odom}")

        particle_filter.update(odom, color)

        post_update_poses.append(particle_filter.get_current_pose())

        if verbose and ground_truth_pose is not None:
            print(f"ground truth pose: {ground_truth_pose}")
            print(f"post update pose: {particle_filter.get_current_pose()}")

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

    return post_update_poses
