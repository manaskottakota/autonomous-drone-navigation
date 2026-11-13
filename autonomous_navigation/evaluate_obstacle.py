"""Evaluate an obstacle PPO policy on seeded wall or randomized scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="PPO model path")
    parser.add_argument("--scenario", choices=("wall", "random"), default="wall")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-history", action="store_true", help="Use model trained with recent command/progress features")
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(scenario=args.scenario, action_history=args.action_history, max_speed=args.speed_limit)
    trials = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            obs, _ = env.reset(seed=seed)
            initial = env.start_position.copy()
            previous = initial.copy()
            path_length, steps = 0.0, 0
            done, info = False, {}
            distance_history, action_history, step_motion = [], [], []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                position = env.aviary._getDroneStateVector(0)[:3]
                distance_history.append(float(info["distance"]))
                action_history.append(np.asarray(action, dtype=float))
                step_motion.append(float(np.linalg.norm(position - previous)))
                path_length += float(np.linalg.norm(position - previous))
                previous = position.copy()
                steps += 1
                done = terminated or truncated
            net_displacement = float(np.linalg.norm(previous - initial))
            if info["is_success"]:
                failure_category = "success"
            elif info["collision"]:
                failure_category = "collision"
            elif info["crashed"]:
                failure_category = "out_of_bounds"
            else:
                tail = max(1, len(action_history) // 4)
                tail_action = float(np.mean(np.linalg.norm(action_history[-tail:], axis=1)))
                tail_motion = float(np.mean(step_motion[-tail:]))
                failure_category = "timeout_low_motion" if tail_action < 0.2 and tail_motion < 0.01 else "timeout_active_nonconvergence"
            trials.append({
                "seed": seed,
                "success": bool(info["is_success"]),
                "collision": bool(info["collision"]),
                "crashed": bool(info["crashed"]),
                "final_distance_m": float(info["distance"]),
                "time_s": steps * env.dt,
                "average_speed_mps": path_length / max(steps * env.dt, 1e-8),
                "path_efficiency": net_displacement / max(path_length, 1e-8),
                "failure_category": failure_category,
                "minimum_goal_distance_m": float(min(distance_history)),
            })
            if (index + 1) % 10 == 0:
                print(f"Evaluated {index + 1}/{args.episodes}")
        successes = [trial for trial in trials if trial["success"]]
        report = {
            "scenario": args.scenario,
            "seeds": {"first": args.seed, "last": args.seed + args.episodes - 1, "episodes": args.episodes},
            "success_rate": float(np.mean([trial["success"] for trial in trials])),
            "collision_rate": float(np.mean([trial["collision"] for trial in trials])),
            "crash_rate": float(np.mean([trial["crashed"] for trial in trials])),
            "mean_completion_time_s": float(np.mean([trial["time_s"] for trial in successes])) if successes else None,
            "mean_average_speed_mps": float(np.mean([trial["average_speed_mps"] for trial in successes])) if successes else None,
            "mean_final_distance_m": float(np.mean([trial["final_distance_m"] for trial in trials])),
            "mean_path_efficiency": float(np.mean([trial["path_efficiency"] for trial in trials])),
            "failure_categories": {
                category: sum(trial["failure_category"] == category for trial in trials)
                for category in sorted({trial["failure_category"] for trial in trials})
            },
            "trials": trials,
        }
        output = args.output or Path(f"autonomous_navigation/results/obstacle_{args.scenario}_ppo.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
        print(f"Saved evaluation to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
