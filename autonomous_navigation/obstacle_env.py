"""LiDAR-observed 3D obstacle navigation with PID motor stabilization."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


@dataclass(frozen=True)
class BoxObstacle:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


WALL_SCENE = {
    "start": (-1.1, 0.0, 0.7),
    "goal": (1.1, 0.0, 0.7),
    "obstacles": (BoxObstacle((0.0, 0.0, 0.75), (0.18, 0.58, 0.75)),),
}


class ObstacleNavigationAviary(gym.Env):
    """Goal navigation using relative goal, velocity, and simulated LiDAR only.

    The policy receives no absolute position or obstacle geometry. Internal
    simulator position is used only to integrate high-level commands into the
    target position consumed by the existing low-level PID controller.
    """

    metadata = {"render_modes": ["human"]}
    AZIMUTH_RAYS = 24
    ELEVATIONS = (-20.0, 0.0, 20.0)
    LIDAR_RAYS = AZIMUTH_RAYS * len(ELEVATIONS)
    LIDAR_RANGE = 3.0

    def __init__(
        self,
        render_mode: str | None = None,
        scenario: str = "random",
        max_episode_steps: int = 200,
        obstacle_count_range: tuple[int, int] = (1, 4),
        action_history: bool = False,
        max_speed: float = 0.45,
        visualize_lidar: bool = True,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if scenario not in ("wall", "random"):
            raise ValueError("scenario must be 'wall' or 'random'")
        self.render_mode = render_mode
        self.scenario = scenario
        self.max_episode_steps = max_episode_steps
        self.set_obstacle_count_range(*obstacle_count_range)
        self.action_history = action_history
        self.visualize_lidar = visualize_lidar
        self.control_hz = 48
        self.action_repeat = 4
        self.dt = self.action_repeat / self.control_hz
        self.set_speed_limit(max_speed)
        self.success_radius = 0.18
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.base_observation_size = 6 + self.LIDAR_RAYS
        obs_size = self.base_observation_size + (5 if self.action_history else 0)
        self.observation_space = spaces.Box(
            low=np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 1.0, dtype=np.float32), dtype=np.float32,
        )
        self.aviary: CtrlAviary | None = None
        self.controller: DSLPIDControl | None = None
        self.obstacles: tuple[BoxObstacle, ...] = ()
        self.obstacle_ids: list[int] = []
        self.start_position = np.zeros(3)
        self.goal = np.zeros(3)
        self.position_target = np.zeros(3)
        self.previous_distance = 0.0
        self.previous_goal_ray_m = self.LIDAR_RANGE
        self.safe_deceleration = 0.65
        self.action_smoothness_weight = 0.05
        self.steps = 0
        self.last_ranges = np.ones(self.LIDAR_RAYS, dtype=np.float32)
        self.debug_ray_ids: list[int] = []
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_distance_progress = 0.0
        self.last_ray_progress = 0.0

    def _close_sim(self) -> None:
        if self.aviary is not None:
            self.aviary.close()
            self.aviary = None
        self.obstacle_ids.clear()
        self.debug_ray_ids.clear()

    def _random_scenario(self):
        start = np.array([
            self.np_random.uniform(-1.1, -0.75),
            self.np_random.uniform(-0.55, 0.55),
            self.np_random.uniform(0.58, 0.82),
        ])
        goal = np.array([
            self.np_random.uniform(0.75, 1.1),
            self.np_random.uniform(-0.55, 0.55),
            self.np_random.uniform(0.58, 0.82),
        ])
        count = int(self.np_random.integers(self.min_obstacles, self.max_obstacles + 1))
        obstacles: list[BoxObstacle] = []
        direction = goal[:2] - start[:2]
        direction /= max(np.linalg.norm(direction), 1e-8)
        perpendicular = np.array([-direction[1], direction[0]])
        for index in range(count):
            if index == 0 or self.np_random.random() < 0.65:
                fraction = self.np_random.uniform(0.3, 0.7)
                xy = start[:2] + fraction * (goal[:2] - start[:2])
                xy += perpendicular * self.np_random.uniform(-0.18, 0.18)
            else:
                xy = self.np_random.uniform([-0.45, -0.8], [0.45, 0.8])
            half = np.array([
                self.np_random.uniform(0.12, 0.25),
                self.np_random.uniform(0.16, 0.36),
                self.np_random.uniform(0.3, 0.8),
            ])
            center = np.array([xy[0], xy[1], self.np_random.uniform(0.45, 0.9)])
            if np.linalg.norm(center[:2] - start[:2]) < half[0] + 0.3:
                center[1] += 0.45
            if np.linalg.norm(center[:2] - goal[:2]) < half[0] + 0.3:
                center[1] -= 0.45
            obstacles.append(BoxObstacle(tuple(center), tuple(half)))
        return start, goal, tuple(obstacles)

    def set_obstacle_count_range(self, minimum: int, maximum: int) -> None:
        """Set the randomized layout difficulty without changing goal sampling."""
        if not 1 <= minimum <= maximum <= 4:
            raise ValueError("obstacle count range must satisfy 1 <= min <= max <= 4")
        self.min_obstacles, self.max_obstacles = minimum, maximum

    def set_speed_limit(self, speed_mps: float) -> None:
        """Set the horizontal high-level command speed in metres per second."""
        if not np.isfinite(speed_mps) or not 0.1 <= speed_mps <= 1.2:
            raise ValueError("speed limit must be finite and between 0.1 and 1.2 m/s")
        self.max_speed = float(speed_mps)

    def _create_obstacles(self) -> None:
        client = self.aviary.getPyBulletClient()
        for obstacle in self.obstacles:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle.half_extents, physicsClientId=client)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=obstacle.half_extents,
                rgbaColor=(0.85, 0.18, 0.12, 1), physicsClientId=client,
            )
            body = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=shape, baseVisualShapeIndex=visual,
                basePosition=obstacle.center, physicsClientId=client,
            )
            self.obstacle_ids.append(body)
        if self.render_mode == "human":
            p.addUserDebugLine(self.start_position, self.goal, (0.1, 0.8, 0.2), lineWidth=1.5,
                                lifeTime=0, physicsClientId=client)
            p.addUserDebugText("goal", self.goal + np.array([0, 0, 0.15]), (0.1, 0.8, 0.2),
                               physicsClientId=client)

    def _sense(self, state: np.ndarray) -> np.ndarray:
        position = state[:3]
        yaw = float(state[9])
        origins, endpoints = [], []
        for elevation_deg in self.ELEVATIONS:
            elevation = np.deg2rad(elevation_deg)
            for ray in range(self.AZIMUTH_RAYS):
                angle = yaw + 2 * np.pi * ray / self.AZIMUTH_RAYS
                direction = np.array([
                    np.cos(elevation) * np.cos(angle),
                    np.cos(elevation) * np.sin(angle),
                    np.sin(elevation),
                ])
                # Offset the ray origin beyond the quadrotor body so rays do not
                # immediately report a hit on the vehicle itself.
                origin = position + 0.12 * direction
                origins.append(origin.tolist())
                endpoints.append((position + self.LIDAR_RANGE * direction).tolist())
        results = p.rayTestBatch(origins, endpoints, physicsClientId=self.aviary.getPyBulletClient())
        self.last_ranges = np.asarray([result[2] for result in results], dtype=np.float32)
        if self.render_mode == "human" and self.visualize_lidar:
            client = self.aviary.getPyBulletClient()
            # Draw the horizontal ring only. The policy still receives all
            # 72 range beams; drawing the other 48 lines adds GUI overhead
            # without making the obstacle-avoidance maneuver easier to read.
            display_indices = range(self.AZIMUTH_RAYS, 2 * self.AZIMUTH_RAYS)
            for display_index, i in enumerate(display_indices):
                origin, endpoint = origins[i], endpoints[i]
                result = results[i]
                hit = result[3] if result[0] >= 0 else endpoint
                color = (1, 0.2, 0.1) if result[0] >= 0 else (0.2, 0.7, 1)
                old_id = self.debug_ray_ids[display_index] if display_index < len(self.debug_ray_ids) else None
                # Keep debug items alive while the simulation runs so their
                # IDs remain valid when PyBullet replaces them on the next
                # control step. A short lifetime makes replaceItemUniqueId
                # fail once the item expires (notably in the macOS GUI build).
                debug_args = {}
                if old_id is not None:
                    debug_args["replaceItemUniqueId"] = old_id
                debug_id = p.addUserDebugLine(origin, hit, color, lineWidth=1,
                                              lifeTime=0, physicsClientId=client,
                                              **debug_args)
                if display_index >= len(self.debug_ray_ids):
                    self.debug_ray_ids.append(debug_id)
                else:
                    self.debug_ray_ids[display_index] = debug_id
        return self.last_ranges

    def _observation(self, state: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        goal_relative = np.clip((self.goal - state[:3]) / 3.0, -1, 1)
        velocity = np.clip(state[10:13] / 2.0, -1, 1)
        observation = [goal_relative, velocity, ranges]
        if self.action_history:
            progress = np.clip([self.last_distance_progress / 0.1, self.last_ray_progress / 1.5], -1, 1)
            observation.extend((self.last_action, progress))
        return np.concatenate(observation).astype(np.float32)

    def _goal_ray_distance(self, state: np.ndarray, ranges: np.ndarray) -> float:
        """Read the horizontal LiDAR beam most aligned with the relative goal."""
        relative = self.goal - state[:3]
        desired_bearing = np.arctan2(relative[1], relative[0])
        relative_bearing = (desired_bearing - float(state[9]) + np.pi) % (2 * np.pi) - np.pi
        ray_index = int(np.round(relative_bearing * self.AZIMUTH_RAYS / (2 * np.pi))) % self.AZIMUTH_RAYS
        return float(ranges[self.AZIMUTH_RAYS + ray_index] * self.LIDAR_RANGE)

    def _speed_adaptive_clearance_penalty(self, clearance_m: float, horizontal_speed_mps: float) -> float:
        """Penalize entering the measured stopping envelope around obstacles."""
        stopping_distance = horizontal_speed_mps**2 / (2.0 * self.safe_deceleration)
        safe_clearance = max(0.45, 0.15 + stopping_distance)
        return 0.8 * max(0.0, safe_clearance - clearance_m)

    def _collided(self) -> bool:
        client = self.aviary.getPyBulletClient()
        drone_id = int(self.aviary.DRONE_IDS[0])
        return any(p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id, physicsClientId=client)
                   for obstacle_id in self.obstacle_ids)

    def _swept_collision(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Catch brief contacts that resolve between PyBullet control frames."""
        padding = 0.08  # approximate horizontal/vertical collision radius of the Crazyflie
        direction = end - start
        for obstacle in self.obstacles:
            center = np.asarray(obstacle.center)
            extent = np.asarray(obstacle.half_extents) + padding
            lower, upper = center - extent, center + extent
            near, far = 0.0, 1.0
            for axis in range(3):
                if abs(direction[axis]) < 1e-9:
                    if start[axis] < lower[axis] or start[axis] > upper[axis]:
                        near, far = 1.0, 0.0
                        break
                    continue
                t1 = (lower[axis] - start[axis]) / direction[axis]
                t2 = (upper[axis] - start[axis]) / direction[axis]
                near = max(near, min(t1, t2))
                far = min(far, max(t1, t2))
            if near <= far and far >= 0.0 and near <= 1.0:
                return True
        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._close_sim()
        if options and "start" in options and "goal" in options and "obstacles" in options:
            start = np.asarray(options["start"], dtype=float)
            goal = np.asarray(options["goal"], dtype=float)
            obstacles = tuple(options["obstacles"])
        elif self.scenario == "wall":
            start = np.asarray(WALL_SCENE["start"], dtype=float)
            goal = np.asarray(WALL_SCENE["goal"], dtype=float)
            obstacles = WALL_SCENE["obstacles"]
        else:
            start, goal, obstacles = self._random_scenario()
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("start and goal must each contain three coordinates")
        if any(not isinstance(obstacle, BoxObstacle) for obstacle in obstacles):
            raise TypeError("obstacles must be BoxObstacle instances")
        self.start_position, self.goal, self.obstacles = start.copy(), goal.copy(), tuple(obstacles)
        self.aviary = CtrlAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=start.reshape(1, 3), initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB, pyb_freq=240, ctrl_freq=self.control_hz,
            gui=self.render_mode == "human", obstacles=False, user_debug_gui=False,
        )
        self.controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        state_batch, _ = self.aviary.reset(seed=seed)
        state = state_batch[0]
        self._create_obstacles()
        self.position_target = state[:3].copy()
        self.steps = 0
        self.previous_distance = float(np.linalg.norm(self.goal - state[:3]))
        self.last_action.fill(0.0)
        self.last_distance_progress = 0.0
        self.last_ray_progress = 0.0
        ranges = self._sense(state)
        self.previous_goal_ray_m = self._goal_ray_distance(state, ranges)
        return self._observation(state, ranges), {"distance": self.previous_distance}

    def step(self, action: np.ndarray):
        if self.aviary is None or self.controller is None:
            raise RuntimeError("call reset() before step()")
        command = np.asarray(action, dtype=np.float32).reshape(3).clip(-1, 1)
        self.position_target += command * np.array([self.max_speed, self.max_speed, 0.18]) * self.dt
        self.position_target = np.clip(self.position_target, [-2, -2, 0.3], [2, 2, 1.8])
        state = None
        collision = False
        client = self.aviary.getPyBulletClient()
        if self.render_mode == "human":
            # PyBullet's GUI otherwise redraws every 240 Hz physics update.
            # The policy and LiDAR only update at 12 Hz, so render once after
            # each four-step PID batch instead.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=client)
        try:
            for _ in range(self.action_repeat):
                prior_position = self.aviary._getDroneStateVector(0)[:3].copy()
                motor_action, _, _ = self.controller.computeControlFromState(
                    control_timestep=self.aviary.CTRL_TIMESTEP,
                    state=self.aviary._getDroneStateVector(0),
                    target_pos=self.position_target, target_rpy=np.zeros(3),
                )
                state_batch, _, _, _, _ = self.aviary.step(motor_action.reshape(1, 4))
                state = state_batch[0]
                collision = self._collided() or self._swept_collision(prior_position, state[:3])
                if collision:
                    break
        except Exception:
            if self.render_mode == "human":
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client)
            raise
        self.steps += 1
        ranges = self._sense(state)
        distance = float(np.linalg.norm(self.goal - state[:3]))
        goal_ray_m = self._goal_ray_distance(state, ranges)
        success = distance <= self.success_radius
        out_of_bounds = np.any(np.abs(state[:2]) > 2.2) or state[2] < 0.12 or state[2] > 2.0
        crashed = bool(collision or out_of_bounds)
        # Potential shaping from the sensor beam along the goal direction gives
        # dense feedback to sidestep a wall and reveal a clear passage.
        ray_progress = np.clip(goal_ray_m, 0.0, 1.5) - np.clip(self.previous_goal_ray_m, 0.0, 1.5)
        horizontal_clearance_m = float(np.min(ranges[self.AZIMUTH_RAYS:2 * self.AZIMUTH_RAYS]) * self.LIDAR_RANGE)
        horizontal_speed = float(np.linalg.norm(state[10:12]))
        proximity_penalty = (
            0.2 * max(0.0, 0.5 - goal_ray_m)
            + self._speed_adaptive_clearance_penalty(horizontal_clearance_m, horizontal_speed)
        )
        command_change_penalty = self.action_smoothness_weight * float(np.sum((command - self.last_action) ** 2))
        goal_capture_progress = max(0.0, 0.5 - self.previous_distance) - max(0.0, 0.5 - distance)
        reward = (
            5.0 * (self.previous_distance - distance)
            + 4.0 * goal_capture_progress
            + 1.5 * ray_progress - 0.02 - proximity_penalty - command_change_penalty
        )
        if success:
            reward += 16.0
        if crashed:
            reward -= 25.0
        self.last_action = command.copy()
        self.last_distance_progress = self.previous_distance - distance
        self.last_ray_progress = ray_progress
        self.previous_distance = distance
        self.previous_goal_ray_m = goal_ray_m
        terminated = bool(success or crashed)
        truncated = self.steps >= self.max_episode_steps
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client)
        return self._observation(state, ranges), float(reward), terminated, truncated, {
            "distance": distance, "is_success": success, "collision": collision,
            "crashed": crashed, "lidar_min_m": float(np.min(ranges) * self.LIDAR_RANGE),
            "goal_ray_m": goal_ray_m,
            "horizontal_clearance_m": horizontal_clearance_m,
        }

    def close(self):
        self._close_sim()
