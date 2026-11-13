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

For a paired comparison on the same randomized scenarios, with new seeds:

```sh
./.venv/bin/python -m autonomous_navigation.compare --episodes 100 --seed 2000
```

The comparison writes `autonomous_navigation/results/ppo_vs_pid.json` and
reports success and crash rates, success time, final distance, and path
efficiency for each controller. Path efficiency is net displacement divided
by distance traveled. It saves per-seed outcomes and paired bootstrap 95%
confidence intervals for time and path-efficiency differences. Both use a 0.15 m
success radius.

Models default to `autonomous_navigation/models/goal_ppo.zip`. Simulation
outputs and generated model artifacts are not source files; preserve the
training seed, timestep count, and evaluation command with each reported run.
Evaluation prints success/crash rates, final distance, elapsed time, and path
efficiency. The current environment has no obstacles; its crash metric counts
ground contact or leaving the flight bounds.

## LiDAR obstacle avoidance

The separate `ObstacleNavigationAviary` uses 72 simulated 3D PyBullet rays,
goal-relative position, and velocity as policy observations. Its PPO action is
an XYZ velocity command converted to a moving target for the DSL PID. The
observation excludes absolute position and obstacle geometry. A fixed wall
scene is available for the first training stage; randomized scenes contain one
to four static boxes.

```sh
./.venv/bin/python -m autonomous_navigation.train_obstacle --scenario wall --timesteps 100000 --seed 17
./.venv/bin/python -m autonomous_navigation.evaluate_obstacle autonomous_navigation/models/obstacle_wall.zip --scenario wall --episodes 20 --seed 10000
./.venv/bin/python -m autonomous_navigation.obstacle_demo --model autonomous_navigation/models/obstacle_wall.zip --scenario wall
```

For randomized multi-obstacle training and evaluation, use `--scenario random`
and hold out evaluation seeds from the training seed range. The evaluator saves
per-trial success, collision, completion time, and path-efficiency results.
