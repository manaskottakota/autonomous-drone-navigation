"""Paired evaluation of LiDAR-only PPO and live-map A* on moving scenes.

The A* reference receives the current simulator obstacle map and replans every
0.5 seconds. It is an oracle-map reference, not a sensor-only baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import BoxObstacle, ObstacleNavigationAviary
from autonomous_navigation.planning import astar_route


def rollout(env: ObstacleNavigationAviary, model: PPO | None, seed: int, replan_steps: int) -> dict:
    obs, _ = env.reset(seed=seed)
    initial = env.start_position.copy()
    previous = initial.copy()
    path_length, steps = 0.0, 0
    waypoint = env.goal.copy()
    done, info = False, {}
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            position = env.aviary._getDroneStateVector(0)[:3]
            if steps % replan_steps == 0 or np.linalg.norm(waypoint - position) < 0.2:
                current_boxes = list(env.obstacles)
                current_boxes.extend(
                    BoxObstacle(tuple(center), obstacle.half_extents)
                    for center, obstacle in zip(env.dynamic_centers, env.dynamic_obstacles)
                )
                route = astar_route(position, env.goal, tuple(current_boxes), cruise_z=float(position[2]))
                if route is None:
                    waypoint = env.goal.copy()
                else:
                    waypoint = next(
                        (point for point in route[1:] if np.linalg.norm(point - position) >= 0.25),
                        route[-1],
                    )
            command_scale = np.array([env.max_speed, env.max_speed, 0.18]) * env.dt
            action = np.clip((waypoint - position) / np.maximum(command_scale, 1e-8), -1, 1).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        position = env.aviary._getDroneStateVector(0)[:3]
        path_length += float(np.linalg.norm(position - previous))
        previous = position.copy()
        steps += 1
        done = terminated or truncated
    return {
        "success": bool(info["is_success"]),
        "collision": bool(info["collision"]),
        "crashed": bool(info["crashed"]),
        "final_distance_m": float(info["distance"]),
        "time_s": steps * env.dt,
        "average_speed_mps": path_length / max(steps * env.dt, 1e-8),
        "path_efficiency": float(np.linalg.norm(previous - initial) / max(path_length, 1e-8)),
    }


def summarize(trials: list[dict]) -> dict:
    successful = [trial for trial in trials if trial["success"]]
    return {
        "success_rate": float(np.mean([trial["success"] for trial in trials])),
        "collision_rate": float(np.mean([trial["collision"] for trial in trials])),
        "mean_completion_time_s": float(np.mean([trial["time_s"] for trial in successful])) if successful else None,
        "mean_average_speed_mps": float(np.mean([trial["average_speed_mps"] for trial in successful])) if successful else None,
        "mean_path_efficiency": float(np.mean([trial["path_efficiency"] for trial in trials])),
        "mean_final_distance_m": float(np.mean([trial["final_distance_m"] for trial in trials])),
        "failure_categories": {
            key: sum(trial["failure_category"] == key for trial in trials)
            for key in sorted({trial["failure_category"] for trial in trials})
        },
    }


def categorize(trial: dict) -> dict:
    if trial["success"]:
        category = "success"
    elif trial["collision"]:
        category = "collision"
    elif trial["crashed"]:
        category = "out_of_bounds"
    else:
        category = "timeout"
    return {**trial, "failure_category": category}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Dynamic-observation PPO model path")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=84000)
    parser.add_argument("--output", type=Path, default=Path("autonomous_navigation/results/dynamic_ppo_vs_live_map_astar.json"))
    parser.add_argument("--action-history", action="store_true")
    parser.add_argument("--policy-hz", type=int, choices=(12, 24, 48), default=12)
    parser.add_argument("--speed-limit", type=float, default=0.90)
    parser.add_argument("--replan-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.episodes <= 0 or args.replan_seconds <= 0:
        parser.error("episodes and replanning interval must be positive")
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(
        scenario="dynamic_random", dynamic_observation=True,
        action_history=args.action_history, policy_hz=args.policy_hz, max_speed=args.speed_limit,
    )
    ppo_trials, astar_trials = [], []
    try:
        replan_steps = max(1, round(args.replan_seconds / env.dt))
        for index in range(args.episodes):
            seed = args.seed + index
            ppo_trials.append(categorize(rollout(env, model, seed, replan_steps)))
            astar_trials.append(categorize(rollout(env, None, seed, replan_steps)))
            if (index + 1) % 10 == 0:
                print(f"Evaluated paired dynamic scenes {index + 1}/{args.episodes}")
    finally:
        env.close()
    report = {
        "seed_range": {"first": args.seed, "last": args.seed + args.episodes - 1, "episodes": args.episodes},
        "ppo_observation": "goal-relative state, velocity, LiDAR ranges/range history; no obstacle geometry",
        "astar_observation": "live PyBullet obstacle map, replanned every %.3f seconds" % args.replan_seconds,
        "policy_hz": args.policy_hz,
        "ppo_lidar_only": summarize(ppo_trials),
        "astar_map_aware": summarize(astar_trials),
        "trials": [
            {"seed": args.seed + i, "ppo": ppo, "astar_map_aware": astar}
            for i, (ppo, astar) in enumerate(zip(ppo_trials, astar_trials))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
    print(f"Saved paired results to {args.output}")


if __name__ == "__main__":
    main()
