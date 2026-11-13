"""Fly one simulated Crazyflie from a start point to a fixed 3D goal.

Run from the repository root with ``./.venv/bin/python -m autonomous_navigation.point_to_point``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


def run(
    gui: bool = True,
    duration: float = 12.0,
    start: np.ndarray | None = None,
    goal: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Run the point-to-point flight and return (success, final_goal_distance)."""
    start = np.array([0.0, 0.0, 0.5]) if start is None else np.asarray(start, dtype=float)
    goal = np.array([1.0, 0.7, 0.8]) if goal is None else np.asarray(goal, dtype=float)
    threshold = 0.15
    control_hz = 48
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=start.reshape(1, 3),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=control_hz,
        gui=gui,
        obstacles=False,
        user_debug_gui=gui,
    )
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((1, 4))
    steps = int(duration * control_hz)
    began = time.time()
    obs, _ = env.reset()
    action[0], _, _ = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=start,
        target_rpy=np.zeros(3),
    )
    if gui:
        client = env.getPyBulletClient()
        import pybullet as p
        p.addUserDebugLine(start, goal, [0, 1, 0], lineWidth=2, lifeTime=0, physicsClientId=client)
        p.addUserDebugText("goal", goal + np.array([0, 0, 0.12]), [0, 0.6, 0], physicsClientId=client)
    try:
        for step in range(steps):
            obs, _, terminated, truncated, _ = env.step(action)
            state = obs[0]
            distance = float(np.linalg.norm(goal - state[0:3]))
            if step % control_hz == 0:
                print(f"t={step / control_hz:4.1f}s position={state[0:3].round(2)} error={distance:.2f}m")
            # Ramp the setpoint to avoid asking the position PID for an
            # instantaneous diagonal acceleration that saturates the motors.
            progress = np.clip(step / (4.0 * control_hz), 0.0, 1.0)
            commanded_goal = start + progress * (goal - start)
            action[0], _, _ = controller.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state,
                target_pos=commanded_goal,
                target_rpy=np.zeros(3),
            )
            if distance <= threshold:
                print(f"Reached goal threshold at {step / control_hz:.2f}s (error={distance:.3f} m)")
                return True, distance
            if terminated or truncated:
                break
            if gui:
                env.render()
                sync(step, began, env.CTRL_TIMESTEP)
        final_distance = float(np.linalg.norm(goal - env._getDroneStateVector(0)[0:3]))
        print(f"Did not reach {threshold:.2f} m threshold in {duration:.1f}s (final error={final_distance:.3f} m)")
        return final_distance <= threshold, final_distance
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration", type=float, default=12.0, help="Flight time limit in seconds")
    args = parser.parse_args()
    success, error = run(gui=args.gui, duration=args.duration)
    print(f"Result: {'SUCCESS' if success else 'FAIL'}; goal error {error:.3f} m")
