"""Evaluate an obstacle PPO policy on seeded wall or randomized scenes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="PPO model path")
    parser.add_argument("--scenario", choices=("wall", "random", "complex", "dynamic_wall", "dynamic_random"), default="wall")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-history", action="store_true", help="Use model trained with recent command/progress features")
    parser.add_argument("--dynamic-observation", action="store_true", help="Use model trained with sensor-only LiDAR history")
    parser.add_argument("--policy-hz", type=int, choices=(12, 24, 48), default=12)
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    parser.add_argument("--max-obstacles", type=int, help="Maximum randomized obstacles (1-12); complex defaults to 10")
    parser.add_argument("--obstacle-speed-min", type=float, default=0.12, help="Minimum randomized dynamic obstacle speed in m/s")
    parser.add_argument("--obstacle-speed-max", type=float, default=0.30, help="Maximum randomized dynamic obstacle speed in m/s")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(scenario=args.scenario, action_history=args.action_history,
                                   dynamic_observation=args.dynamic_observation, policy_hz=args.policy_hz,
                                   max_speed=args.speed_limit)
    if args.max_obstacles is not None:
        if not 1 <= args.max_obstacles <= 12:
            parser.error("--max-obstacles must be between 1 and 12")
        env.set_obstacle_count_range(1, args.max_obstacles)
    env.set_dynamic_speed_range(args.obstacle_speed_min, args.obstacle_speed_max)
    trials = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            obs, _ = env.reset(seed=seed)
            initial = env.start_position.copy()
            previous = initial.copy()
            path_length, steps = 0.0, 0
            done, info = False, {}
            distance_history, action_history, step_motion, latencies = [], [], [], []
            sensor_alert_time = None
            while not done:
                infer_started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                latencies.append(time.perf_counter() - infer_started)
                if args.dynamic_observation and sensor_alert_time is None:
                    rate_start = env.base_observation_size + (5 if args.action_history else 0)
                    rates = np.abs(obs[rate_start:rate_start + env.LIDAR_RAYS]) * 2.0 * env.LIDAR_RANGE
                    if np.any(rates > 0.12):
                        sensor_alert_time = steps * env.dt
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
                "first_range_rate_alert_s": sensor_alert_time,
                "obstacle_count": len(env.obstacles) + len(env.dynamic_obstacles),
                "geometry_types": sorted({type(item).__name__ for item in env.obstacles}),
                "policy_inference_median_ms": float(np.median(latencies) * 1000.0),
                "policy_inference_p95_ms": float(np.percentile(latencies, 95) * 1000.0),
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
            "policy_hz": env.policy_hz,
            "median_inference_ms": float(np.median([trial["policy_inference_median_ms"] for trial in trials])),
            "p95_inference_ms": float(np.percentile([trial["policy_inference_p95_ms"] for trial in trials], 95)),
            "mean_first_range_rate_alert_s": float(np.mean([
                trial["first_range_rate_alert_s"] for trial in trials
                if trial["first_range_rate_alert_s"] is not None
            ])) if any(trial["first_range_rate_alert_s"] is not None for trial in trials) else None,
            "failure_categories": {
                category: sum(trial["failure_category"] == category for trial in trials)
                for category in sorted({trial["failure_category"] for trial in trials})
            },
            "by_obstacle_count": {
                str(count): {
                    "episodes": sum(t["obstacle_count"] == count for t in trials),
                    "success_rate": float(np.mean([t["success"] for t in trials if t["obstacle_count"] == count])),
                    "collision_rate": float(np.mean([t["collision"] for t in trials if t["obstacle_count"] == count])),
                }
                for count in sorted({t["obstacle_count"] for t in trials})
            },
            "by_geometry": {
                geometry: {
                    "episodes": sum(geometry in t["geometry_types"] for t in trials),
                    "success_rate": float(np.mean([t["success"] for t in trials if geometry in t["geometry_types"]])),
                    "collision_rate": float(np.mean([t["collision"] for t in trials if geometry in t["geometry_types"]])),
                }
                for geometry in sorted({g for t in trials for g in t["geometry_types"]})
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
