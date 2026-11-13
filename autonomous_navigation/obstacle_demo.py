"""Show a trained LiDAR-observing PPO policy navigating the fixed wall scene."""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="autonomous_navigation/models/obstacle_wall.zip")
    parser.add_argument("--scenario", choices=("wall", "random"), default="wall")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--hold-seconds", type=float, default=4.0)
    parser.add_argument("--action-history", action="store_true", help="Use model trained with recent command/progress features")
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(render_mode="human", scenario=args.scenario, action_history=args.action_history,
                                   max_speed=args.speed_limit)
    try:
        obs, _ = env.reset(seed=args.seed)
        done, info = False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time.sleep(env.dt)
        print(f"success={info['is_success']} collision={info['collision']} distance={info['distance']:.3f}m")
        time.sleep(max(0.0, args.hold_seconds))
    finally:
        env.close()


if __name__ == "__main__":
    main()
