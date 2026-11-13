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
./.venv/bin/python -m autonomous_navigation.train_obstacle --scenario wall --timesteps 50000 --seed 17
./.venv/bin/python -m autonomous_navigation.evaluate_obstacle autonomous_navigation/models/obstacle_wall.zip --scenario wall --episodes 20 --seed 10000
./.venv/bin/python -m autonomous_navigation.obstacle_demo --model autonomous_navigation/models/obstacle_wall.zip --scenario wall
```

For randomized multi-obstacle training and evaluation, use `--scenario random`
and hold out evaluation seeds from the training seed range. The evaluator saves
per-trial success, collision, completion time, and path-efficiency results.

```sh
./.venv/bin/python -m autonomous_navigation.train_obstacle --scenario random --timesteps 100000 --seed 17 --output autonomous_navigation/models/obstacle_random
./.venv/bin/python -m autonomous_navigation.compare_obstacles autonomous_navigation/models/obstacle_random.zip --episodes 100 --seed 30000 --training-seed 17
```

The fixed-wall model reached 20/20 held-out seeds after 50,176 PPO steps. The
random-layout model was trained to 501,760 total steps and reached 68/100
held-out scenes (seeds 30000–30099) with zero collisions. The map-aware A* route
planner reached 100/100 with zero collisions, 5.82 s mean completion time, and
0.639 mean path efficiency. PPO's successful episodes averaged 5.86 s and 0.564
path efficiency. Its XY grid assumes constant cruise altitude and is given the
obstacle map; PPO only receives LiDAR, relative goal position, and velocity.
The randomized PPO result is a working baseline, not yet a reliable replacement
for planning.

## High-speed navigation (Milestone 4)

The obstacle environment exposes a configurable horizontal target-motion speed
limit (default 0.45 m/s). This limits how quickly the PPO command moves the PID
position setpoint; it is not a hard cap on physical velocity. The policy acts
at 12 Hz, the DSL PID updates at 48 Hz, and PyBullet advances at 240 Hz.

An isolated straight-line pulse measured a 0.492 m/s peak at the 0.45 m/s
setting, 1.02 m/s² peak acceleration, and 0.193 m / 0.67 s to brake below
0.05 m/s. At 0.90 m/s, the same pulse reached 0.987 m/s and 2.07 m/s², and
required 0.440 m / 1.42 s to brake below 0.05 m/s. The reward therefore
increases its LiDAR clearance margin with measured horizontal speed and adds a
small penalty for abrupt command changes.

Train a new policy from the preserved sensor-history checkpoint with four
speed/obstacle curriculum stages (0.45, 0.60, 0.75, and 0.90 m/s):

```sh
./.venv/bin/python -m autonomous_navigation.train_obstacle \
  --scenario random \
  --resume autonomous_navigation/models/obstacle_history_gamma995_seeded29.zip \
  --action-history --speed-curriculum --timesteps 600000 --seed 43 \
  --output autonomous_navigation/models/obstacle_speed_curriculum_seed43
```

The command saves per-stage checkpoints and leaves the source model intact.
The 0.90 m/s model reached 93% on 200 untouched randomized scenes (seeds
90000–90199), with a 0.5% collision rate, 6.42 s mean successful completion
time, 0.872 m/s average speed, and 0.352 path efficiency. A* found a route in
all 200 scenes but its fixed-speed waypoint follower reached only 81.5% at
0.90 m/s; it is not a valid high-speed upper bound without corner-speed
adaptation. Full paired metrics and per-episode outcomes are saved in
`results/obstacle_speed_curriculum_heldout_90000_90199.json`.

Validation on seeds 70000–70099 compared the previous checkpoint at 0.45 m/s
against the new checkpoint at two limits. The old model achieved 92% success,
2% collisions, and 0.546 m/s average speed. The new model achieved 90% success
with zero collisions at 0.75 m/s (0.735 m/s average), and 94% success with 1%
collisions at 0.90 m/s (0.874 m/s average). Efficiency fell from 0.556 to
0.374–0.365, so the faster policy trades away directness. The 0.90 m/s setting
was selected for the final untouched run; it is not collision-free.

Before retraining, runtime speed scaling of the previous checkpoint was probed
on the same 100 scenes (seeds 60000–60099). At 0.45 / 0.60 / 0.75 m/s command
limits, success was 84% / 92% / 95%, collision rate was 1% / 3% / 3%, and
average speed was 0.538 / 0.684 / 0.819 m/s. This showed that simply raising
the command limit increased speed but did not preserve the collision rate.
Those per-speed comparisons remain in `results/obstacle_speed_probe_*.json`.

Run the trained visual wall demo with the horizontal LiDAR overlay, or hide the
overlay with `--no-lidar`. All 72 rays remain active for the policy; the GUI
draws only the horizontal 24-ray ring. It batches four 240 Hz physics updates
between redraws and avoids the upstream text-only render output. On an Apple
M1 Pro, measured playback improved from about 0.4× to 0.59× real time; it is
still slower than real time.

```sh
./.venv/bin/python -m autonomous_navigation.obstacle_demo \
  --model autonomous_navigation/models/obstacle_speed_curriculum_seed43.zip \
  --scenario wall --action-history --speed-limit 0.90
```
