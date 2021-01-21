import tempfile
import pytest
import numpy as np
from scipy.stats import circstd
from ..simulator import make_map, make_input_data
from ..filter import CarpetBasedParticleFilter, load_input_log, OdomMeasurement, Pose
from ..carpet_map import generate_random_map
from ..colors import LIGHT_BLUE, UNCLASSIFIED

plot = False

if plot:
    from ..visualisation import plot_map, plot_particles, plot_pose


def test_filter_perfect_data():

    carpet = make_map()
    simulated_data = make_input_data(
        odom_pos_noise_std_dev=0,
        odom_heading_noise_std_dev=0,
        color_noise=0,
    )

    np.random.seed(123)
    particle_filter = CarpetBasedParticleFilter(carpet)
    for odom, color, ground_truth_pose in simulated_data:
        particle_filter.update(odom, color)

        if plot:
            plot_map(carpet, show=False)
            plot_particles(particle_filter._pfilter.particles, show=False)
            estimated_pose = particle_filter.get_current_pose()
            plot_pose(
                estimated_pose,
                color="red",
                show=False,
            )
            plot_pose(ground_truth_pose)

    estimated_pose = particle_filter.get_current_pose()
    pos_tol = 0.5  # meters
    rot_tol = 0.5  # radians

    def wrap_angle(x):
        return np.mod(x + np.pi, 2 * np.pi) - np.pi

    assert ground_truth_pose.x == pytest.approx(estimated_pose.x, abs=pos_tol)
    assert ground_truth_pose.y == pytest.approx(estimated_pose.y, abs=pos_tol)
    angle_diff = wrap_angle(ground_truth_pose.heading - estimated_pose.heading)
    assert angle_diff == pytest.approx(0, abs=rot_tol)


def test_log_inputs():
    """
    check that a filter can be configured to log inputs
    """

    input_data = make_input_data(
        odom_pos_noise_std_dev=0,
        odom_heading_noise_std_dev=0,
        color_noise=0,
    )

    carpet = make_map()

    # run the particle filter over the input data while logging inputs
    particle_filter = CarpetBasedParticleFilter(carpet, log_inputs=True)

    for odom, color, ground_truth_pose in input_data:
        particle_filter.update(odom, color, ground_truth=ground_truth_pose)

    # save the logged inputs and confirm they match the actual inputs
    with tempfile.TemporaryDirectory() as tmpdirname:

        log_file = f"{tmpdirname}/filter_input_log.pickle"
        particle_filter.write_input_log(log_file)

        logged_inputs = load_input_log(log_file)

    assert logged_inputs == input_data


def test_init():
    """
    Particle filter initialisation:
    Filter should initialise on first update, to a state where most particles are
    located on tiles of the given color
    """
    np.random.seed(123)
    carpet = make_map()
    particle_filter = CarpetBasedParticleFilter(carpet)

    color = LIGHT_BLUE

    particle_filter._most_recent_color = LIGHT_BLUE
    particle_filter._pfilter_init()

    particles = particle_filter.get_particles()

    color_of_tiles_under_the_particles = carpet.get_color_at_coords(
        particles[:, 0:2])

    # copying constants out of filter.py
    # TODO: implement parameterisation
    # in filter constructor
    WEIGHT_FN_P = 0.95
    N_PARTICLES = 500

    assert sum(color_of_tiles_under_the_particles ==
               LIGHT_BLUE.index) / N_PARTICLES == pytest.approx(WEIGHT_FN_P,
                                                                abs=0.01)


def test_seed():
    """
    Start particle filter in large map with randomly distributed particles.
    Seed filter, confirm that particles are distributed around the seed pose
    """
    np.random.seed(123)
    carpet = generate_random_map(shape=(100, 100), cell_size=0.5)

    particle_filter = CarpetBasedParticleFilter(carpet)

    particle_filter.update(odom=OdomMeasurement(dx=0, dy=0, dheading=0),
                           color=UNCLASSIFIED)

    seed_pose = Pose(x=25, y=25, heading=np.pi)
    pos_tol = 2
    heading_tol = 0.3
    pos_std_dev = 1.
    heading_std_dev = 0.3

    def position_difference(p1: Pose, p2: Pose):
        return np.sqrt(
            np.sum((np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))**2))

    def angular_difference(p1: Pose, p2: Pose):
        return np.abs((np.mod(p1.heading - p2.heading + np.pi, 2 * np.pi)) -
                      np.pi)

    def particles_in_tollerance() -> bool:
        particles = particle_filter.get_particles()
        pose = particle_filter.get_current_pose()

        return (position_difference(pose, seed_pose) < pos_tol
                and angular_difference(pose, seed_pose) < heading_tol
                and np.std(particles[:, 0]) < pos_std_dev * 1.1
                and np.std(particles[:, 1]) < pos_std_dev * 1.1
                and circstd(particles[:, 2]) < heading_std_dev * 1.1)

    # initially expect particles spread out
    assert not particles_in_tollerance()

    # after seeding, particles should be distributed closely around
    # the seed position

    particle_filter.seed(seed_pose,
                         pos_std_dev=pos_std_dev,
                         heading_std_dev=heading_std_dev)

    assert particles_in_tollerance()
    if plot:
        plot_map(carpet, show=False)
        plot_particles(particle_filter._pfilter.particles, show=False)
        estimated_pose = particle_filter.get_current_pose()
        plot_pose(
            estimated_pose,
            color="red",
            show=False,
        )
        plot_pose(seed_pose)
