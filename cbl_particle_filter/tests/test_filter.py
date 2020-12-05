from ..simulator import make_map, make_input_data
from ..filter import CarpetBasedParticleFilter
def test_filter():
    carpet = make_map()
    simulated_data = make_input_data()

    particle_filter = CarpetBasedParticleFilter(carpet)
    for odom, color, ground_truth_pose in simulated_data:
        particle_filter.update(odom, color)

        # TODO: assert pose is near - does not need to be exactly equal
        assert particle_filter.current_pose == ground_truth_pose