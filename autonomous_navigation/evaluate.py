"""Evaluate a saved PPO policy over seeded, randomized point-to-point trials."""
from __future__ import annotations

import argparse
import numpy as np
from stable_baselines3 import PPO

from autonomous_navigation.env import PointToPointAviary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="PPO model path (without or with .zip)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    env = PointToPointAviary()
    model = PPO.load(args.model, device="cpu")
    successes, crashes, distances, lengths, times, efficiencies = 0, 0, [], [], [], []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done, steps, info = False, 0, {}
            previous_position = env.start_position.copy()
            path_length = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                position = env.aviary._getDroneStateVector(0)[:3]
                path_length += float(np.linalg.norm(position - previous_position))
                previous_position = position.copy()
                steps += 1
                done = terminated or truncated
            successes += int(info.get("is_success", False))
            crashes += int(info.get("crashed", False))
            distances.append(info["distance"])
            lengths.append(steps)
            times.append(steps * env.dt)
            direct_distance = float(np.linalg.norm(env.goal - env.start_position))
            efficiencies.append(direct_distance / max(path_length, direct_distance, 1e-8))
            if (episode + 1) % 10 == 0:
                print(f"Evaluated {episode + 1}/{args.episodes}")
        print(f"success_rate={successes / args.episodes:.3f} ({successes}/{args.episodes})")
        print(f"crash_rate={crashes / args.episodes:.3f}; mean_final_distance_m={np.mean(distances):.3f}")
        print(f"mean_time_s={np.mean(times):.2f}; mean_path_efficiency={np.mean(efficiencies):.3f}; mean_steps={np.mean(lengths):.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
