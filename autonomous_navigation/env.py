"""Goal-reaching Gymnasium task with velocity commands and PID motor control."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def sample_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a start/goal pair with the same seeded distribution as reset()."""
    rng, _ = seeding.np_random(seed)
    return _sample_scenario(rng)


def _sample_scenario(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.5, 0.8)])
    offset = rng.uniform(-0.8, 0.8, size=3)
    if np.linalg.norm(offset) < 0.4:
        offset[0] += 0.5
    goal = start + offset
    goal[2] = np.clip(goal[2], 0.45, 1.2)
    return start, goal


class PointToPointAviary(gym.Env):
    """Single-drone navigation task; actions are body-independent world XYZ velocity."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 160):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_hz = 48
        self.action_repeat = 4
        self.dt = self.action_repeat / self.control_hz
        self.max_speed = 0.45
        self.success_radius = 0.15
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Relative goal (xyz, normalized by 2m) and world velocity (normalized by 2m/s).
        self.observation_space = spaces.Box(
            low=np.full(6, -1.0, dtype=np.float32),
            high=np.full(6, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.aviary: CtrlAviary | None = None
        self.controller: DSLPIDControl | None = None
        self.position_target = np.zeros(3)
        self.start_position = np.zeros(3)
        self.goal = np.zeros(3)
        self.steps = 0
        self.previous_distance = 0.0

    def _close_sim(self) -> None:
        if self.aviary is not None:
            self.aviary.close()
            self.aviary = None

    def _observation(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate(((self.goal - state[:3]) / 2.0, state[10:13] / 2.0)).clip(-1, 1).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._close_sim()
        if options and "start" in options and "goal" in options:
            start = np.asarray(options["start"], dtype=float)
            self.goal = np.asarray(options["goal"], dtype=float)
        else:
            start, self.goal = _sample_scenario(self.np_random)
        if start.shape != (3,) or self.goal.shape != (3,):
            raise ValueError("start and goal must each contain three coordinates")
        self.start_position = start.copy()
        self.aviary = CtrlAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=start.reshape(1, 3), initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB, pyb_freq=240, ctrl_freq=self.control_hz,
            gui=self.render_mode == "human", obstacles=False, user_debug_gui=False,
        )
        self.controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        state_batch, _ = self.aviary.reset(seed=seed)
        state = state_batch[0]
        self.position_target = state[:3].copy()
        self.steps = 0
        self.previous_distance = float(np.linalg.norm(self.goal - state[:3]))
        return self._observation(state), {"distance": self.previous_distance}

    def step(self, action: np.ndarray):
        if self.aviary is None or self.controller is None:
            raise RuntimeError("call reset() before step()")
        command = np.asarray(action, dtype=np.float32).reshape(3).clip(-1, 1)
        self.position_target += command * self.max_speed * self.dt
        # Keep the PID setpoint within the navigation volume and prevent ground commands.
        self.position_target = np.clip(self.position_target, [-2, -2, 0.3], [2, 2, 1.8])
        state = None
        for _ in range(self.action_repeat):
            motor_action, _, _ = self.controller.computeControlFromState(
                control_timestep=self.aviary.CTRL_TIMESTEP,
                state=self.aviary._getDroneStateVector(0),
                target_pos=self.position_target,
                target_rpy=np.zeros(3),
            )
            state_batch, _, _, _, _ = self.aviary.step(motor_action.reshape(1, 4))
            state = state_batch[0]
        self.steps += 1
        distance = float(np.linalg.norm(self.goal - state[:3]))
        success = distance <= self.success_radius
        crashed = state[2] < 0.12 or np.any(np.abs(state[:2]) > 2.2)
        reward = 5.0 * (self.previous_distance - distance) - 0.02 - 0.01 * float(np.dot(command, command))
        if success:
            reward += 5.0
        if crashed:
            reward -= 5.0
        self.previous_distance = distance
        terminated = bool(success or crashed)
        truncated = self.steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.aviary.render()
        return self._observation(state), float(reward), terminated, truncated, {
            "distance": distance, "is_success": success, "crashed": crashed,
        }

    def close(self):
        self._close_sim()
