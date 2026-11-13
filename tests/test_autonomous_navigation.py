"""Contract checks for the project-specific point-to-point environment."""
import numpy as np

from autonomous_navigation.env import PointToPointAviary
from autonomous_navigation.env import sample_scenario
from autonomous_navigation.obstacle_env import BoxObstacle, ObstacleNavigationAviary
from autonomous_navigation.obstacle_env import WALL_SCENE
from autonomous_navigation.planning import astar_route


def test_reset_observation_and_seeded_scenario():
    env = PointToPointAviary(max_episode_steps=4)
    try:
        obs, info = env.reset(seed=31)
        first_start, first_goal = env.start_position.copy(), env.goal.copy()
        sampled_start, sampled_goal = sample_scenario(31)
        np.testing.assert_allclose(first_start, sampled_start)
        np.testing.assert_allclose(first_goal, sampled_goal)
        assert env.observation_space.contains(obs)
        assert np.isfinite(info["distance"])
        obs, _ = env.reset(seed=31)
        np.testing.assert_allclose(env.start_position, first_start)
        np.testing.assert_allclose(env.goal, first_goal)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_action_reward_and_goal_termination():
    env = PointToPointAviary(max_episode_steps=4)
    try:
        obs, _ = env.reset(seed=9, options={"start": [0.0, 0.0, 0.6], "goal": [0.0, 0.0, 0.6]})
        assert env.observation_space.contains(obs)
        assert env.action_space.contains(np.zeros(3, dtype=np.float32))
        next_obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert env.observation_space.contains(next_obs)
        assert np.isfinite(reward)
        assert terminated and not truncated
        assert info["is_success"]
        assert not info["crashed"]
    finally:
        env.close()


def test_wall_scene_lidar_observation_and_collision_detection():
    env = ObstacleNavigationAviary(scenario="wall", max_episode_steps=4)
    try:
        obs, _ = env.reset(seed=2)
        assert env.observation_space.contains(obs)
        assert obs.shape == (6 + env.LIDAR_RAYS,)
        assert np.any(obs[6:] < 0.95), "wall should be detected by at least one LiDAR ray"
        assert obs.size == 78, "policy input should contain goal-relative state and LiDAR rays only"
    finally:
        env.close()


def test_obstacle_contact_terminates_episode():
    env = ObstacleNavigationAviary(scenario="wall", max_episode_steps=4)
    try:
        start = np.array([0.0, 0.0, 0.7])
        obs, _ = env.reset(seed=3, options={
            "start": start, "goal": [1.0, 0.0, 0.7],
            "obstacles": (BoxObstacle((0.0, 0.0, 0.7), (0.3, 0.3, 0.3)),),
        })
        _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert terminated
        assert info["collision"]
        assert info["crashed"]
    finally:
        env.close()


def test_map_aware_planner_routes_around_fixed_wall():
    route = astar_route(
        np.asarray(WALL_SCENE["start"], dtype=float),
        np.asarray(WALL_SCENE["goal"], dtype=float),
        WALL_SCENE["obstacles"],
        cruise_z=0.7,
    )
    assert route is not None
    np.testing.assert_allclose(route[0], WALL_SCENE["start"])
    np.testing.assert_allclose(route[-1], WALL_SCENE["goal"])
    assert len(route) >= 3, "route should bend around the wall rather than cut through it"
