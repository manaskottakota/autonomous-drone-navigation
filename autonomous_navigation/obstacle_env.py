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
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if scenario not in ("wall", "random"):
            raise ValueError("scenario must be 'wall' or 'random'")
        self.render_mode = render_mode
        self.scenario = scenario
        self.max_episode_steps = max_episode_steps
        self.control_hz = 48
        self.action_repeat = 4
        self.dt = self.action_repeat / self.control_hz
        self.max_speed = 0.45
        self.success_radius = 0.18
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        obs_size = 6 + self.LIDAR_RAYS
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
        self.steps = 0
        self.last_ranges = np.ones(self.LIDAR_RAYS, dtype=np.float32)
        self.debug_ray_ids: list[int] = []

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
        count = int(self.np_random.integers(1, 5))
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
        if self.render_mode == "human":
            client = self.aviary.getPyBulletClient()
            for i, ((origin, endpoint), result) in enumerate(zip(zip(origins, endpoints), results)):
                hit = result[3] if result[0] >= 0 else endpoint
                color = (1, 0.2, 0.1) if result[0] >= 0 else (0.2, 0.7, 1)
                old_id = self.debug_ray_ids[i] if i < len(self.debug_ray_ids) else -1
                # Keep debug items alive while the simulation runs so their
                # IDs remain valid when PyBullet replaces them on the next
                # control step. A short lifetime makes replaceItemUniqueId
                # fail once the item expires (notably in the macOS GUI build).
                debug_id = p.addUserDebugLine(origin, hit, color, lineWidth=1,
                                              lifeTime=0, replaceItemUniqueId=old_id,
                                              physicsClientId=client)
                if i >= len(self.debug_ray_ids):
                    self.debug_ray_ids.append(debug_id)
                else:
                    self.debug_ray_ids[i] = debug_id
        return self.last_ranges

    def _observation(self, state: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        goal_relative = np.clip((self.goal - state[:3]) / 3.0, -1, 1)
        velocity = np.clip(state[10:13] / 2.0, -1, 1)
        return np.concatenate((goal_relative, velocity, ranges)).astype(np.float32)

    def _goal_ray_distance(self, state: np.ndarray, ranges: np.ndarray) -> float:
        """Read the horizontal LiDAR beam most aligned with the relative goal."""
        relative = self.goal - state[:3]
        desired_bearing = np.arctan2(relative[1], relative[0])
        relative_bearing = (desired_bearing - float(state[9]) + np.pi) % (2 * np.pi) - np.pi
        ray_index = int(np.round(relative_bearing * self.AZIMUTH_RAYS / (2 * np.pi))) % self.AZIMUTH_RAYS
        return float(ranges[self.AZIMUTH_RAYS + ray_index] * self.LIDAR_RANGE)

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
        proximity_penalty = (
            0.2 * max(0.0, 0.5 - goal_ray_m)
            + 0.8 * max(0.0, 0.45 - horizontal_clearance_m)
        )
        reward = 5.0 * (self.previous_distance - distance) + 1.5 * ray_progress - 0.02 - proximity_penalty
        if success:
            reward += 8.0
        if crashed:
            reward -= 25.0
        self.previous_distance = distance
        self.previous_goal_ray_m = goal_ray_m
        terminated = bool(success or crashed)
        truncated = self.steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.aviary.render()
        return self._observation(state, ranges), float(reward), terminated, truncated, {
            "distance": distance, "is_success": success, "collision": collision,
            "crashed": crashed, "lidar_min_m": float(np.min(ranges) * self.LIDAR_RANGE),
            "goal_ray_m": goal_ray_m,
            "horizontal_clearance_m": horizontal_clearance_m,
        }

    def close(self):
        self._close_sim()
