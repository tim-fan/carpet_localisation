# cbl_particle_filter

A particle filter implementing carpet-based localisation.

Inputs to particle filter are:
* odometry: pose delta since previous update
* color: current detected carpet color under robot
* map: grid map of carpet colors

Output of particle filter is the current pose (2D) of the robot

## TODO:
* Have created some simulated data for testing - now need to actually implement the filter
* add basic visualisation for map and particles

