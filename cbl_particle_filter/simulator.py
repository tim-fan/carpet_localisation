"""simulator.
Generates simulated inputs for testing the particle filter
"""

from typing import List, Tuple
import numpy as np
from .carpet_map import CarpetMap
from .filter import OdomMeasurement, Pose
from .colors import Color, COLORS, color_from_index


def make_map() -> CarpetMap:
    return CarpetMap(
        grid=np.array([
            [0, 0, 0, 2, 1],
            [0, 0, 2, 2, 2],  # yapf: disable
            [0, 1, 3, 2, 1],
            [1, 1, 1, 1, 1]
        ]),
        cell_size=0.5)


def make_input_data(
    odom_pos_noise_std_dev: float = 0.02,
    odom_heading_noise_std_dev: float = 0.005,
    color_noise:
    float = 0.2,  # percent chance of making error in color classification
) -> List[Tuple[OdomMeasurement, Color, Pose]]:
    """
    Simulate a circular drive.
    Start at x= 1.25, y=0.25, heading = 0
    drive a 0.8m radius circle to the right, 5 loops

    Returns list of odom and color measurements, and associated ground truth poses
    """

    # first generate ground truth trajectory
    # then derive measurements
    # then add noise

    np.random.seed(1234)

    carpet = make_map()

    num_loops = 5
    radius = 0.8
    num_updates = 100

    heading_ground_truth = np.linspace(0,
                                       np.pi * 2 * num_loops,
                                       num=num_updates)
    x_ground_truth = 1.25 + np.sin(heading_ground_truth) * radius
    y_ground_truth = 0.25 + radius - np.cos(heading_ground_truth) * radius

    # for odometry, determine *robot-frame* pose deltas per update
    odom_x_ground_truth = np.diff(
        x_ground_truth,
        append=x_ground_truth[-1]) * np.cos(heading_ground_truth) + np.diff(
            y_ground_truth,
            append=y_ground_truth[-1]) * np.sin(heading_ground_truth)
    odom_y_ground_truth = -np.diff(x_ground_truth, append=x_ground_truth[
        -1]) * np.sin(heading_ground_truth) + np.diff(
            y_ground_truth,
            append=y_ground_truth[-1]) * np.cos(heading_ground_truth)
    odom_heading_ground_truth = np.diff(heading_ground_truth,
                                        append=heading_ground_truth[-1])

    # add noise to odom
    odom_x = odom_x_ground_truth + np.random.normal(
        scale=odom_pos_noise_std_dev, size=num_updates)
    odom_y = odom_y_ground_truth + np.random.normal(
        scale=odom_pos_noise_std_dev, size=num_updates)
    odom_heading = odom_heading_ground_truth + np.random.normal(
        scale=odom_heading_noise_std_dev, size=num_updates)

    odom_measurements = [
        OdomMeasurement(
            dx=odom_x[i],
            dy=odom_y[i],
            dheading=odom_heading[i],
        ) for i in range(num_updates)
    ]

    # get ground truth colors:
    color_ground_truth = carpet.get_color_at_coords(
        np.column_stack([x_ground_truth, y_ground_truth]))

    color_indices = sorted([c.index for c in COLORS])
    n_colors = len(COLORS)
    assert np.all(np.unique(carpet.grid) == color_indices)

    correct_choice_prob = 1 - color_noise
    false_choice_prob = color_noise / (n_colors - 1)

    color_errors = np.random.choice(color_indices,
                                    size=num_updates,
                                    p=[
                                        correct_choice_prob, false_choice_prob,
                                        false_choice_prob, false_choice_prob
                                    ])
    colors = np.mod(color_ground_truth + color_errors, n_colors)

    color_measurements = [color_from_index[c] for c in colors]

    ground_truth_poses = [
        Pose(
            x=x_ground_truth[i],
            y=y_ground_truth[i],
            heading=heading_ground_truth[i],
        ) for i in range(num_updates)
    ]

    return [(odom, color, ground_truth_pose) for odom, color, ground_truth_pose
            in zip(odom_measurements, color_measurements, ground_truth_poses)]
