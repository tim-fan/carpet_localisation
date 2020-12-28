import pytest
import numpy as np
from ..simulator import make_map, make_input_data
from ..filter import CarpetBasedParticleFilter


def test_filter_perfect_data():
    plot = False

    if plot:
        from ..visualisation import plot_map, plot_particles, plot_pose

    carpet = make_map()
    simulated_data = make_input_data(
        odom_pos_noise_std_dev=0,
        odom_heading_noise_std_dev=0,
        color_noise=0,
    )

    particle_filter = CarpetBasedParticleFilter(carpet)
    for odom, color, ground_truth_pose in simulated_data:
        particle_filter.update(odom, color)

        if plot:
            plot_map(carpet, show=False)
            plot_particles(particle_filter._pfilter.particles, show=False)
            estimated_pose = particle_filter.get_current_pose()
            plot_pose(
                estimated_pose.x,
                estimated_pose.y,
                estimated_pose.heading,
                color="red",
                show=False,
            )
            plot_pose(
                ground_truth_pose.x,
                ground_truth_pose.y,
                ground_truth_pose.heading,
            )

    estimated_pose = particle_filter.get_current_pose()
    pos_tol = 0.5  # meters
    rot_tol = 0.5  # radians

    def wrap_angle(x):
        return np.mod(x + np.pi, 2 * np.pi) - np.pi

    assert ground_truth_pose.x == pytest.approx(estimated_pose.x, abs=pos_tol)
    assert ground_truth_pose.y == pytest.approx(estimated_pose.y, abs=pos_tol)
    angle_diff = wrap_angle(ground_truth_pose.heading - estimated_pose.heading)
    assert angle_diff == pytest.approx(0, abs=rot_tol)
