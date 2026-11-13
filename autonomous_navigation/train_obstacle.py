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
    parser.add_argument("--resume", type=Path, help="Resume training from a saved PPO checkpoint")
    parser.add_argument("--curriculum", action="store_true", help="Progress from 1 to 2 to 4 obstacles")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    env = ObstacleNavigationAviary(scenario=args.scenario)
    if args.check_env:
        check_env(env, warn=True)
        print("Obstacle environment API check passed")
        env.close()
        return
    base_env = env
    env = Monitor(base_env)
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed, n_steps=512, batch_size=64,
            learning_rate=3e-4, gamma=args.gamma, device="cpu",
        )
    output = args.output or Path(f"autonomous_navigation/models/obstacle_{args.scenario}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.curriculum and args.scenario == "random":
        phases = ((1, 0.25), (2, 0.25), (4, 0.50))
        completed = 0
        for phase, (max_obstacles, fraction) in enumerate(phases, start=1):
            phase_steps = args.timesteps - completed if phase == len(phases) else round(args.timesteps * fraction)
            if phase_steps <= 0:
                continue
            base_env.set_obstacle_count_range(1, max_obstacles)
            print(f"Curriculum phase {phase}: 1-{max_obstacles} obstacles, {phase_steps} steps")
            model.learn(total_timesteps=phase_steps, reset_num_timesteps=args.resume is None and completed == 0)
            completed += phase_steps
            model.save(f"{output}_stage_{phase}")
    else:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=args.resume is None)
    model.save(output)
    print(f"Saved policy to {output}.zip")
    env.close()


if __name__ == "__main__":
    main()
