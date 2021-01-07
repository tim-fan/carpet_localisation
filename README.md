# cbl_particle_filter

A particle filter implementing carpet-based localisation.

Inputs to particle filter are:
* odometry: pose delta since previous update
* color: current detected carpet color under robot
* map: grid map of carpet colors

Output of particle filter is the current pose (2D) of the robot

![simulation_playback](docs/source/_static/simulation_playback.gif)

Figure: playback of particle filter using simulated input data (see [simulator.py](./cbl_particle_filter/simulator.py)), driving in a loop on a small map. Legend:
* green arrow: ground truth pose
* large red arrow: estimated pose from particle filter
* small red arrows: particle filter particles
* blue/grey/beige background: carpet color pattern map

## Dev setup:

```
git clone git@github.com:tim-fan/carpet_localisation.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run tests
```bash
pytest .
```
Or to view output of plotting tests:
```bash
pytest . --show_plot True
```