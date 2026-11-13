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

## Dynamic obstacle avoidance (Milestone 5)

Dynamic boxes are PyBullet collision bodies whose poses advance at the 48 Hz
PID/control rate. Bounded trajectories include perpendicular crossings and
signed same-direction/head-on motion. Randomized scenes mix static and moving
boxes and vary mover speeds. Collision checking uses relative swept motion
between physics updates so a box cannot pass through the drone between samples.
Existing static scenes and checkpoints remain unchanged.

The dynamic PPO observation can append 72 normalized LiDAR range-rate values
computed from consecutive scans. It contains no obstacle coordinates, body IDs,
or velocities. Its legacy input prefix can be initialized from a successful
static policy while zero-initializing weights for new sensor history:

```sh
./.venv/bin/python -m autonomous_navigation.train_obstacle \
  --scenario dynamic_wall --dynamic-observation --action-history --policy-hz 12 \
  --transfer-from autonomous_navigation/models/obstacle_speed_curriculum_seed43.zip \
  --transfer-only --output autonomous_navigation/models/dynamic_transfer_baseline_seed43
./.venv/bin/python -m autonomous_navigation.train_obstacle \
  --scenario dynamic_wall --dynamic-observation --action-history --policy-hz 12 \
  --resume autonomous_navigation/models/dynamic_transfer_baseline_seed43.zip \
  --dynamic-curriculum --gamma 0.995 --learning-rate 0.00003 \
  --timesteps 120000 --seed 551 \
  --output autonomous_navigation/models/dynamic_finetune_low_lr_seed551
```

The curriculum starts with one slow moving box, then advances to 1–4 mixed
obstacles, obstacle speeds from 0.12 to 0.40 m/s, and 0.45–0.90 m/s command
limits. PPO decision frequency is configurable at 12/24/48 Hz; PID remains at
48 Hz and PyBullet physics at 240 Hz. The default episode duration stays at
16.67 s at each policy rate (400 steps at 24 Hz). On this Mac, policy inference
measured 0.12 ms median and 0.31 ms p95 at 12 Hz (an 83 ms budget), and 0.12 ms
median / 0.30 ms p95 at 24 Hz (a 41.7 ms budget). Physics and batched LiDAR
rays already run inside PyBullet's native code, so profiling did not identify
a useful C++ extension to maintain.

Held-out results at a 0.90 m/s command limit:

- Across 200 dynamic scenes with crossing, head-on, and same-direction movers
  at 0.12–0.30 m/s (seeds 87000–87199), the zero-shot transferred checkpoint
  achieved 92.5% success and 1.5% collisions, with 7.24 s mean successful time,
  0.856 m/s path speed, and 0.328 path efficiency.
- A low-learning-rate PPO fine-tune (120,000 steps at 3e-5) achieved 92.0%
  success and 2.5% collisions at 12 Hz on a separate 200-scene set (seeds
  88000–88199, movers up to 0.30 m/s). On the same seeds the transferred
  checkpoint achieved 90.5% and 3.0% collisions; the fine-tune had 10
  fine-tune-only successes versus 7 baseline-only successes, faster completion
  (6.48 vs 7.12 s), and better path efficiency (0.359 vs 0.324).
- For movers up to 0.40 m/s, the 12 Hz fine-tune achieved 92.0% success and
  4.0% collisions on seeds 89000–89199. The transferred checkpoint achieved
  88.5% and 3.5% on the same seeds: range-history inputs improved completion
  and path efficiency, but did not lower collision risk.
- On a paired 100-scene 0.40 m/s mover set (seeds 90000–90099), raising policy
  decisions from 12 to 24 Hz improved the transferred policy from 92% success
  / 2% collisions to 98% / 1%, with similar successful completion time (7.48
  vs 7.54 s) and better efficiency (0.328 vs 0.303). On 200 final scenes at
  24 Hz (seeds 90000–90199), it reached 95% with 3.5% collisions, 7.29 s mean
  successful time, and 0.337 path efficiency. The 100-scene result was an
  interim subset; use the 200-scene result for the final estimate. Its first
  sensor-derived LiDAR range-rate threshold alert averaged 0.105 s; that is
  sensor detection timing, not an independently measured evasive reaction time.
- A separate high-learning-rate fine-tune degraded to 89.5% success / 4.5%
  collisions on crossing-only seeds 84000–84199. Fresh PPO from random
  initialization reached 0/100 on its initial held-out set. Transfer learning
  was important; range-history fine-tuning has not consistently lowered
  collision risk below the simpler transferred policy.
- On paired seeds 84000–84099, PPO achieved 91% success with 2% collisions.
  Live-map A* replanning every 0.5 seconds achieved 72% success with 27%
  collisions. A* has privileged current obstacle geometry, but this snapshot
  planner does not predict motion and is not an upper bound.
- Fresh PPO training from random initialization did not learn reliably in the
  120,000-step run: it achieved 0/100 successes on its randomized held-out set.
  Transfer initialization was important for stable goal reaching.
- A static-map A* reachability audit found routes in all 200 sampled scenes
  with seeds 20000–20199. This checks geometric reachability, not every
  time-dependent interaction.

Per-episode reports remain in `autonomous_navigation/results/` and model files
in `autonomous_navigation/models/`; both are ignored by Git. Run paired dynamic
PPO/live-map A* evaluation with:

```sh
./.venv/bin/python -m autonomous_navigation.compare_dynamic \
  autonomous_navigation/models/dynamic_transfer_baseline_seed43.zip \
  --action-history --policy-hz 12 --speed-limit 0.90 --episodes 100 --seed 84000
```

## Complex multi-obstacle courses (Milestone 6, static geometry stage)

The `complex` scenario extends the existing 72-ray/PID environment with 5–10
randomized static obstacles. Geometry includes boxes/walls, vertical cylinders
and thin pillars, spheres, and gates built from three independent PyBullet
primitives so their opening is real and visible to LiDAR. PyBullet supplies the
ray intersections and rigid-body contacts; swept collision checks use
conservative primitive bounds. Random layouts keep obstacle extents separated
and reserve an outer flight corridor, while a map-aware A* audit checks that
each generated scene has a route. The policy still sees only relative goal,
velocity, and LiDAR (plus optional action history); no obstacle map is added.

First test the new scene and its reachability, then fine-tune the preserved
high-speed PPO checkpoint through a 1-to-10-obstacle curriculum:

```sh
./.venv/bin/python -m autonomous_navigation.train_obstacle \
  --scenario complex --complex-curriculum --action-history \
  --resume autonomous_navigation/models/obstacle_speed_curriculum_seed43.zip \
  --gamma 0.995 --learning-rate 0.00003 --timesteps 300000 --seed 61201 \
  --output autonomous_navigation/models/complex_curriculum_seed61201
./.venv/bin/python -m autonomous_navigation.evaluate_obstacle \
  autonomous_navigation/models/complex_curriculum_seed61201.zip \
  --scenario complex --action-history --policy-hz 12 --speed-limit 0.90 \
  --episodes 200 --seed 206000
```

For a visual course, use `obstacle_demo --scenario complex --obstacles 10` with
the saved model. The current checkpoint predates this geometry distribution;
the 20-scene seed-205000 smoke benchmark before complex training was only 55%
successful, with 15% collisions and 30% active timeouts. Treat this as an
initial generalization baseline, not a final performance claim. In the initial
generator audit, all 200 seeds 200000–200199 had A* routes; the layouts
contained 5–10 objects and all four geometry types appeared. This stage adds
static multi-shape courses; generalizing the dynamic motion model over these
geometry types remains a later Milestone 6 substage.

For the moving-wall visualization, launch an interactive macOS Terminal, then
run:

```sh
cd ~/gym-pybullet-drones
source .venv/bin/activate
./.venv/bin/python -m autonomous_navigation.obstacle_demo \
  --model autonomous_navigation/models/dynamic_finetune_low_lr_seed551.zip \
  --scenario dynamic_wall --dynamic-observation --action-history \
  --policy-hz 24 --speed-limit 0.90 --seed 86000
```

The GUI shows moving boxes in amber, the horizontal LiDAR ring, and status text
with speed and nearest range. A headless PyBullet recording with projected
LiDAR rays is available at `results/dynamic_wall_demo.mp4`; the low-rate
fine-tuned policy reached the goal without collision in 3.79 s at 0.848 m/s.
Regenerate it with:

```sh
./.venv/bin/python -m autonomous_navigation.record_dynamic_demo \
  --model autonomous_navigation/models/dynamic_finetune_low_lr_seed551.zip \
  --scenario dynamic_wall --action-history --policy-hz 24 --speed-limit 0.90 \
  --seed 86000 --output autonomous_navigation/results/dynamic_wall_demo.mp4
```

A background GUI launch aborted with macOS `SIGABRT` during AppKit application
registration. The PyBullet DIRECT-rendered video works; the interactive GUI
command still needs direct confirmation from a regular macOS Terminal.
