"""Paired PPO versus PID-only benchmark on identical seeded scenarios."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.env import PointToPointAviary, sample_scenario
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


@dataclass
class Trial:
    success: bool
    crashed: bool
    final_distance_m: float
    time_s: float
    path_efficiency: float


def run_pid(start: np.ndarray, goal: np.ndarray, duration: float = 12.0) -> Trial:
    """Run the four-second-ramped position target through PID only."""
    hz = 48
    radius = 0.15
    with redirect_stdout(io.StringIO()):
        env = CtrlAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=start.reshape(1, 3), initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB, pyb_freq=240, ctrl_freq=hz, gui=False,
            obstacles=False, user_debug_gui=False,
        )
        state_batch, _ = env.reset()
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    initial = state_batch[0]
    action, _, _ = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP, state=initial,
        target_pos=start, target_rpy=np.zeros(3),
    )
    previous = initial[:3].copy()
    path_length = 0.0
    success, crashed = False, False
    elapsed = duration
    final_distance = float(np.linalg.norm(goal - initial[:3]))
    try:
        for step in range(int(duration * hz)):
            state_batch, _, _, _, _ = env.step(action.reshape(1, 4))
            state = state_batch[0]
            position = state[:3]
            path_length += float(np.linalg.norm(position - previous))
            previous = position.copy()
            final_distance = float(np.linalg.norm(goal - position))
            elapsed = (step + 1) / hz
            if final_distance <= radius:
                success = True
                break
            if position[2] < 0.12 or np.any(np.abs(position[:2]) > 2.2):
                crashed = True
                break
            progress = np.clip(step / (4.0 * hz), 0.0, 1.0)
            commanded_goal = start + progress * (goal - start)
            action, _, _ = controller.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP, state=state,
                target_pos=commanded_goal, target_rpy=np.zeros(3),
            )
        net_displacement = float(np.linalg.norm(previous - start))
        efficiency = net_displacement / max(path_length, 1e-8)
        return Trial(success, crashed, final_distance, elapsed, efficiency)
    finally:
        env.close()


def run_ppo(env: PointToPointAviary, model: PPO, seed: int) -> Trial:
    with redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    previous = env.start_position.copy()
    path_length = 0.0
    info = {"distance": float(np.linalg.norm(env.goal - previous)), "is_success": False, "crashed": False}
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        position = env.aviary._getDroneStateVector(0)[:3]
        path_length += float(np.linalg.norm(position - previous))
        previous = position.copy()
        steps += 1
        done = terminated or truncated
    net_displacement = float(np.linalg.norm(previous - env.start_position))
    return Trial(
        bool(info["is_success"]), bool(info["crashed"]), float(info["distance"]),
        steps * env.dt, net_displacement / max(path_length, 1e-8),
    )


def summarize(trials: list[Trial]) -> dict[str, float]:
    successful_times = [trial.time_s for trial in trials if trial.success]
    return {
        "success_rate": float(np.mean([trial.success for trial in trials])),
        "crash_rate": float(np.mean([trial.crashed for trial in trials])),
        "mean_final_distance_m": float(np.mean([trial.final_distance_m for trial in trials])),
        "mean_time_s_successes": float(np.mean(successful_times)) if successful_times else float("nan"),
        "mean_path_efficiency": float(np.mean([trial.path_efficiency for trial in trials])),
    }


def paired_mean_ci(differences: np.ndarray, seed: int = 19) -> dict[str, float]:
    """Bootstrap a percentile interval for a paired mean difference."""
    if len(differences) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(differences)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="autonomous_navigation/models/goal_ppo.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("autonomous_navigation/results/ppo_vs_pid.json"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model = PPO.load(args.model, device="cpu")
    env = PointToPointAviary()
    pid_trials, ppo_trials = [], []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            start, goal = sample_scenario(seed)
            pid_trials.append(run_pid(start, goal))
            ppo_trials.append(run_ppo(env, model, seed))
            if (index + 1) % 10 == 0:
                print(f"Evaluated paired scenarios {index + 1}/{args.episodes}")
    finally:
        env.close()
    results = {
        "seeds": {"first": args.seed, "last": args.seed + args.episodes - 1, "episodes": args.episodes},
        "success_radius_m": 0.15,
        "ppo": summarize(ppo_trials),
        "pid_only": summarize(pid_trials),
        "paired_differences_ppo_minus_pid": {
            "time_s_successes": paired_mean_ci(np.array([
                p.time_s - b.time_s for p, b in zip(ppo_trials, pid_trials) if p.success and b.success
            ])),
            "path_efficiency": paired_mean_ci(np.array([p.path_efficiency - b.path_efficiency for p, b in zip(ppo_trials, pid_trials)])),
        },
        "trials": [
            {"seed": args.seed + i, "ppo": asdict(ppo), "pid_only": asdict(pid)}
            for i, (ppo, pid) in enumerate(zip(ppo_trials, pid_trials))
        ],
    }
    results["ppo_minus_pid_success_rate"] = results["ppo"]["success_rate"] - results["pid_only"]["success_rate"]
    results["ppo_minus_pid_time_s"] = results["ppo"]["mean_time_s_successes"] - results["pid_only"]["mean_time_s_successes"]
    results["ppo_minus_pid_path_efficiency"] = results["ppo"]["mean_path_efficiency"] - results["pid_only"]["mean_path_efficiency"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if key != "trials"}, indent=2))
    print(f"Saved paired benchmark to {args.output}")


if __name__ == "__main__":
    main()
