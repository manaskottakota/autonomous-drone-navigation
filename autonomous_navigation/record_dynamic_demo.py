"""Record a headless PyBullet dynamic-obstacle run with projected LiDAR rays."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from autonomous_navigation.obstacle_env import ObstacleNavigationAviary


def _project(point: np.ndarray, view: np.ndarray, projection: np.ndarray, width: int, height: int):
    clip = projection @ view @ np.append(point, 1.0)
    if clip[3] <= 1e-8:
        return None
    ndc = clip[:3] / clip[3]
    return int((ndc[0] + 1.0) * 0.5 * width), int((1.0 - ndc[1]) * 0.5 * height)


def _draw_line(image: np.ndarray, start: tuple[int, int] | None, end: tuple[int, int] | None, color):
    if start is None or end is None:
        return
    x0, y0 = start
    x1, y1 = end
    count = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, count)).astype(int)
    ys = np.rint(np.linspace(y0, y1, count)).astype(int)
    visible = (xs >= 0) & (xs < image.shape[1]) & (ys >= 0) & (ys < image.shape[0])
    image[ys[visible], xs[visible]] = color


def _capture(env: ObstacleNavigationAviary, width: int, height: int) -> np.ndarray:
    client = env.aviary.getPyBulletClient()
    state = env.aviary._getDroneStateVector(0)
    view_values = p.computeViewMatrix(
        cameraEyePosition=[0.0, -4.2, 3.2], cameraTargetPosition=[0.0, 0.0, 0.72],
        cameraUpVector=[0.0, 0.0, 1.0], physicsClientId=client,
    )
    projection_values = p.computeProjectionMatrixFOV(
        fov=58.0, aspect=width / height, nearVal=0.1, farVal=12.0,
    )
    frame = p.getCameraImage(
        width, height, viewMatrix=view_values, projectionMatrix=projection_values,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client,
    )[2]
    image = np.asarray(frame, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
    view = np.asarray(view_values, dtype=float).reshape(4, 4, order="F")
    projection = np.asarray(projection_values, dtype=float).reshape(4, 4, order="F")
    yaw = float(state[9])
    for ray in range(env.AZIMUTH_RAYS):
        angle = yaw + 2.0 * np.pi * ray / env.AZIMUTH_RAYS
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        origin = state[:3] + 0.12 * direction
        measured_range = float(env.last_ranges[env.AZIMUTH_RAYS + ray]) * env.LIDAR_RANGE
        ray_distance = min(env.LIDAR_RANGE, 0.12 + measured_range * (env.LIDAR_RANGE - 0.12) / env.LIDAR_RANGE)
        endpoint = origin + ray_distance * direction
        start_px = _project(origin, view, projection, width, height)
        end_px = _project(endpoint, view, projection, width, height)
        color = (255, 70, 45) if measured_range < env.LIDAR_RANGE - 0.03 else (55, 195, 255)
        _draw_line(image, start_px, end_px, color)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="autonomous_navigation/models/dynamic_transfer_baseline_seed43.zip")
    parser.add_argument("--scenario", choices=("dynamic_wall", "dynamic_random"), default="dynamic_wall")
    parser.add_argument("--seed", type=int, default=86000)
    parser.add_argument("--output", type=Path, default=Path("autonomous_navigation/results/dynamic_wall_demo.mp4"))
    parser.add_argument("--action-history", action="store_true")
    parser.add_argument("--speed-limit", type=float, default=0.90)
    parser.add_argument("--policy-hz", type=int, choices=(12, 24, 48), default=12)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to encode the recording")
    model = PPO.load(args.model, device="cpu")
    env = ObstacleNavigationAviary(
        scenario=args.scenario, dynamic_observation=True, action_history=args.action_history,
        policy_hz=args.policy_hz, max_speed=args.speed_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    path_length = 0.0
    try:
        obs, _ = env.reset(seed=args.seed)
        previous = env.start_position.copy()
        with tempfile.TemporaryDirectory(prefix="dynamic-drone-") as temporary:
            raw_path = Path(temporary) / "frames.rgb"
            with raw_path.open("wb") as raw:
                raw.write(_capture(env, args.width, args.height).tobytes())
                done, info = False, {}
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    position = env.aviary._getDroneStateVector(0)[:3]
                    path_length += float(np.linalg.norm(position - previous))
                    previous = position.copy()
                    raw.write(_capture(env, args.width, args.height).tobytes())
                    frame_count += 1
                    done = terminated or truncated
            subprocess.run([
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{args.width}x{args.height}", "-framerate", str(env.policy_hz),
                "-i", str(raw_path), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.output),
            ], check=True)
        duration = frame_count * env.dt
        print(
            f"success={info['is_success']} collision={info['collision']} "
            f"distance={info['distance']:.3f}m avg_speed={path_length / max(duration, 1e-8):.3f}m/s "
            f"duration={duration:.2f}s frames={frame_count} output={args.output}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
