"""Train a PPO obstacle-avoidance policy with range-sensor observations."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def transfer_policy_state(target_state: dict, source_state: dict) -> dict:
    """Copy a compatible policy, zero-initializing newly added input features."""
    transferred = {key: value.clone() for key, value in target_state.items()}
    for key, source in source_state.items():
        if key not in transferred:
            continue
        target = transferred[key]
        if target.shape == source.shape:
            transferred[key] = source.clone()
        elif (key.endswith(".0.weight") and target.ndim == source.ndim == 2
              and target.shape[0] == source.shape[0] and target.shape[1] > source.shape[1]):
            target.zero_()
            target[:, :source.shape[1]] = source
            transferred[key] = target
        else:
            raise ValueError(f"cannot transfer mismatched policy parameter {key}: {source.shape} -> {target.shape}")
    return transferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("wall", "random", "dynamic_wall", "dynamic_random"), default="wall")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume training from a saved PPO checkpoint")
    parser.add_argument("--transfer-from", type=Path, help="Initialize from a compatible static policy; added LiDAR-history input weights start at zero")
    parser.add_argument("--transfer-only", action="store_true", help="Save the transferred policy without training")
    parser.add_argument("--curriculum", action="store_true", help="Progress from 1 to 2 to 4 obstacles")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    parser.add_argument("--action-history", action="store_true", help="Include previous command and sensor-derived progress")
    parser.add_argument("--dynamic-observation", action="store_true", help="Append sensor-only LiDAR range-rate history for moving objects")
    parser.add_argument("--policy-hz", type=int, choices=(12, 24, 48), default=12, help="PPO decision frequency; PID remains 48 Hz and physics 240 Hz")
    parser.add_argument("--dynamic-curriculum", action="store_true", help="Train from one slow crossing obstacle to randomized mixed dynamic layouts")
    parser.add_argument("--speed-limit", type=float, default=0.45, help="Horizontal command speed limit in m/s")
    parser.add_argument("--speed-curriculum", action="store_true", help="Increase speed from 0.45 to 0.90 m/s while increasing obstacle count")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    if args.resume and args.transfer_from:
        parser.error("--resume and --transfer-from are mutually exclusive")
    if args.transfer_only and not args.transfer_from:
        parser.error("--transfer-only requires --transfer-from")
    env = ObstacleNavigationAviary(scenario=args.scenario, action_history=args.action_history,
                                   dynamic_observation=args.dynamic_observation, policy_hz=args.policy_hz,
                                   max_speed=args.speed_limit)
    if args.check_env:
        check_env(env, warn=True)
        print("Obstacle environment API check passed")
        env.close()
        return
    base_env = env
    # SB3 seeds its policy RNG, but Gymnasium's own np_random would otherwise
    # be entropy-seeded on its first reset, making scene generation irreproducible.
    base_env.reset(seed=args.seed)
    base_env.close()
    env = Monitor(base_env)
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        model.set_random_seed(args.seed)
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed, n_steps=512, batch_size=64,
            learning_rate=3e-4, gamma=args.gamma, device="cpu",
        )
        if args.transfer_from:
            source = PPO.load(args.transfer_from, device="cpu")
            model.policy.load_state_dict(transfer_policy_state(
                model.policy.state_dict(), source.policy.state_dict(),
            ))
            print(f"Initialized policy weights from {args.transfer_from}; sensor-history weights start at zero")
    output = args.output or Path(f"autonomous_navigation/models/obstacle_{args.scenario}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.transfer_only:
        model.save(output)
        env.close()
        print(f"Saved transferred policy to {output}.zip")
        return
    if args.speed_curriculum and args.scenario == "random":
        phases = ((0.45, 1, 0.20), (0.60, 2, 0.20), (0.75, 3, 0.30), (0.90, 4, 0.30))
        completed = 0
        for phase, (speed, max_obstacles, fraction) in enumerate(phases, start=1):
            phase_steps = args.timesteps - completed if phase == len(phases) else round(args.timesteps * fraction)
            if phase_steps <= 0:
                continue
            base_env.set_speed_limit(speed)
            base_env.set_obstacle_count_range(1, max_obstacles)
            print(f"Speed curriculum phase {phase}: {speed:.2f} m/s, 1-{max_obstacles} obstacles, {phase_steps} steps")
            model.learn(total_timesteps=phase_steps, reset_num_timesteps=args.resume is None and completed == 0)
            completed += phase_steps
            model.save(f"{output}_speed_stage_{phase}")
    elif args.dynamic_curriculum:
        if not args.dynamic_observation:
            parser.error("--dynamic-curriculum requires --dynamic-observation")
        phases = (
            ("dynamic_wall", 0.45, 1, 0.20),
            ("dynamic_random", 0.60, 2, 0.25),
            ("dynamic_random", 0.75, 3, 0.30),
            ("dynamic_random", 0.90, 4, 0.25),
        )
        completed = 0
        for phase, (scenario, speed, max_obstacles, fraction) in enumerate(phases, start=1):
            phase_steps = args.timesteps - completed if phase == len(phases) else round(args.timesteps * fraction)
            if phase_steps <= 0:
                continue
            base_env.scenario = scenario
            base_env.set_speed_limit(speed)
            base_env.set_obstacle_count_range(1, max_obstacles)
            print(f"Dynamic curriculum phase {phase}: {scenario}, {speed:.2f} m/s, 1-{max_obstacles} boxes, {phase_steps} steps")
            model.learn(total_timesteps=phase_steps, reset_num_timesteps=args.resume is None and completed == 0)
            completed += phase_steps
            model.save(f"{output}_dynamic_stage_{phase}")
    elif args.curriculum and args.scenario in ("random", "dynamic_random"):
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
