# Autonomous navigation, milestone 2+

This project-specific package builds on the upstream `CtrlAviary` physics and
`DSLPIDControl`; it does not modify the simulator dynamics. The first stage is
one fixed point-to-point target without obstacles or learning.

From the repository root, use the repository virtual environment explicitly:

```sh
./.venv/bin/python -m autonomous_navigation.point_to_point
./.venv/bin/python -m autonomous_navigation.point_to_point --no-gui
```

The GUI demo draws the goal line and prints position/error as it flies. A
headless run prints the same progress and exits with SUCCESS when it enters the
0.15 m goal radius.

The DSL PID controller uses SciPy's `[x, y, z, w]` quaternion convention, the
same ordering returned by PyBullet. The controller had reordered quaternion
components, so the attitude loop targeted the wrong orientation. Fixing that
and ramping position setpoints over four seconds prevents the overshoot and
crash seen with an immediate diagonal goal command.

## PPO

The Gymnasium action is a bounded XYZ velocity command. The environment
integrates it into a moving position setpoint; the existing DSL PID controller
converts that setpoint into motor RPMs. Observations contain relative goal
position and world velocity. Training uses CPU PPO and a default seed of 7.
Episodes randomize start and goal positions within the obstacle-free flight
volume.

```sh
./.venv/bin/python -m autonomous_navigation.train --check-env
./.venv/bin/python -m autonomous_navigation.train --timesteps 100000 --seed 7
./.venv/bin/python -m autonomous_navigation.evaluate autonomous_navigation/models/goal_ppo.zip --episodes 100 --seed 1000
```

Models default to `autonomous_navigation/models/goal_ppo.zip`. Simulation
outputs and generated model artifacts are not source files; preserve the
training seed, timestep count, and evaluation command with each reported run.
Evaluation prints success/crash rates, final distance, elapsed time, and path
efficiency. The current environment has no obstacles; its crash metric counts
ground contact or leaving the flight bounds.
