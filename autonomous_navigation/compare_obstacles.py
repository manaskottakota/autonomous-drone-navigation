"""Compare sensor-only PPO with map-aware A* on identical held-out scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary
from autonomous_navigation.planning import astar_route


def rollout(env, model, seed: int, waypoints: list[np.ndarray] | None = None, debug: bool = False) -> dict:
    obs, _ = env.reset(seed=seed)
    start = env.start_position.copy()
    previous = start.copy()
    path_length, steps, target_index = 0.0, 0, 1
    done, info = False, {}
    while not done:
        if waypoints is None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            position = env.aviary._getDroneStateVector(0)[:3]
            while target_index < len(waypoints) - 1 and np.linalg.norm(waypoints[target_index] - position) < 0.18:
                target_index += 1
            delta = waypoints[target_index] - position
            command_scale = np.array([env.max_speed, env.max_speed, 0.18]) * env.dt
            action = np.clip(delta / command_scale, -1, 1).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        position = env.aviary._getDroneStateVector(0)[:3]
        path_length += float(np.linalg.norm(position - previous))
        previous = position.copy()
        steps += 1
        done = terminated or truncated
        if debug and (steps % 5 == 0 or done):
            print("planner", steps, "pos", np.round(position, 2), "waypoint", target_index,
                  np.round(waypoints[target_index], 2), "collision", info["collision"])
    return {
        "success": bool(info["is_success"]),
        "collision": bool(info["collision"]),
        "final_distance_m": float(info["distance"]),
        "time_s": steps * env.dt,
        "average_speed_mps": path_length / max(steps * env.dt, 1e-8),
        "path_efficiency": float(np.linalg.norm(previous - start) / max(path_length, 1e-8)),
    }


def summarize(trials: list[dict]) -> dict[str, float]:
    successful = [trial for trial in trials if trial["success"]]
    return {
        "success_rate": float(np.mean([trial["success"] for trial in trials])),
        "collision_rate": float(np.mean([trial["collision"] for trial in trials])),
        "mean_completion_time_s": float(np.mean([trial["time_s"] for trial in successful])) if successful else float("nan"),
        "mean_average_speed_mps": float(np.mean([trial["average_speed_mps"] for trial in successful])) if successful else float("nan"),
        "mean_final_distance_m": float(np.mean([trial["final_distance_m"] for trial in trials])),
        "mean_path_efficiency": float(np.mean([trial["path_efficiency"] for trial in trials])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="autonomous_navigation/models/obstacle_random.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--training-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("autonomous_navigation/results/obstacle_ppo_vs_astar.json"))
    parser.add_argument("--action-history", action="store_true", help="Use model trained with recent command/progress features")
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(scenario="random", action_history=args.action_history, max_speed=args.speed_limit)
    ppo_trials, astar_trials, no_route = [], [], 0
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            env.reset(seed=seed)
            start, goal, obstacles = env.start_position.copy(), env.goal.copy(), env.obstacles
            waypoints = astar_route(start, goal, obstacles, cruise_z=start[2])
            env.close()
            ppo_trials.append(rollout(env, model, seed))
            if waypoints is None:
                no_route += 1
                astar_trials.append({"success": False, "collision": False, "final_distance_m": float(np.linalg.norm(goal-start)),
                                     "time_s": env.max_episode_steps * env.dt, "average_speed_mps": 0.0,
                                     "path_efficiency": 0.0})
            else:
                astar_trials.append(rollout(env, model=None, seed=seed, waypoints=waypoints))
            if (index + 1) % 10 == 0:
                print(f"Evaluated paired scenes {index + 1}/{args.episodes}")
    finally:
        env.close()
    report = {
        "training_seed": args.training_seed,
        "seeds": {"first": args.seed, "last": args.seed + args.episodes - 1, "episodes": args.episodes},
        "ppo": summarize(ppo_trials),
        "astar_map_aware": summarize(astar_trials),
        "astar_no_route_count": no_route,
        "trials": [{"seed": args.seed + i, "ppo": ppo, "astar_map_aware": astar}
                   for i, (ppo, astar) in enumerate(zip(ppo_trials, astar_trials))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
