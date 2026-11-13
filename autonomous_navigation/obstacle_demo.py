"""Show a trained LiDAR-observing PPO policy navigating the fixed wall scene."""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="autonomous_navigation/models/obstacle_wall.zip")
    parser.add_argument("--scenario", choices=("wall", "random", "dynamic_wall", "dynamic_random"), default="wall")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--hold-seconds", type=float, default=4.0)
    parser.add_argument("--action-history", action="store_true", help="Use model trained with recent command/progress features")
    parser.add_argument("--dynamic-observation", action="store_true", help="Use model trained with sensor-only LiDAR history")
    parser.add_argument("--policy-hz", type=int, choices=(12, 24, 48), default=12)
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    parser.add_argument("--no-lidar", action="store_true", help="Hide the optional horizontal LiDAR overlay")
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(render_mode="human", scenario=args.scenario, action_history=args.action_history,
                                   dynamic_observation=args.dynamic_observation, policy_hz=args.policy_hz,
                                   max_speed=args.speed_limit, visualize_lidar=not args.no_lidar)
    try:
        obs, _ = env.reset(seed=args.seed)
        began = time.perf_counter()
        done, info = False, {}
        path_length = 0.0
        previous_position = env.aviary._getDroneStateVector(0)[:3].copy()
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            position = env.aviary._getDroneStateVector(0)[:3].copy()
            path_length += float((sum((position - previous_position) ** 2)) ** 0.5)
            previous_position = position
            steps += 1
            done = terminated or truncated
            time.sleep(env.dt)
        elapsed = time.perf_counter() - began
        sim_time = steps * env.dt
        print(
            f"success={info['is_success']} collision={info['collision']} "
            f"distance={info['distance']:.3f}m avg_speed={path_length / max(sim_time, 1e-8):.3f}m/s "
            f"playback={sim_time / max(elapsed, 1e-8):.2f}x"
        )
        time.sleep(max(0.0, args.hold_seconds))
    finally:
        env.close()


if __name__ == "__main__":
    main()
