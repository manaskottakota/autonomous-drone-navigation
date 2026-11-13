"""Train a PPO obstacle-avoidance policy with range-sensor observations."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("wall", "random"), default="wall")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    env = ObstacleNavigationAviary(scenario=args.scenario)
    if args.check_env:
        check_env(env, warn=True)
        print("Obstacle environment API check passed")
        env.close()
        return
    env = Monitor(env)
    model = PPO(
        "MlpPolicy", env, verbose=1, seed=args.seed, n_steps=512, batch_size=64,
        learning_rate=3e-4, gamma=0.99, device="cpu",
    )
    model.learn(total_timesteps=args.timesteps)
    output = args.output or Path(f"autonomous_navigation/models/obstacle_{args.scenario}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved policy to {output}.zip")
    env.close()


if __name__ == "__main__":
    main()
