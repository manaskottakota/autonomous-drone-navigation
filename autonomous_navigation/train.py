"""Train a PPO goal-reaching policy; PID remains the low-level flight controller."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from autonomous_navigation.env import PointToPointAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("autonomous_navigation/models/goal_ppo"))
    parser.add_argument("--check-env", action="store_true", help="Run SB3's API checker without training")
    args = parser.parse_args()
    env = PointToPointAviary()
    if args.check_env:
        check_env(env, warn=True)
        print("Environment API check passed")
        env.close()
        return
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=512, batch_size=64,
                learning_rate=3e-4, gamma=0.99, device="cpu")
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved policy to {args.output}.zip")
    env.close()


if __name__ == "__main__":
    main()
