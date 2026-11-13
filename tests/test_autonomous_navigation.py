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


def test_obstacle_curriculum_range_and_reachable_seeded_scenes():
    env = ObstacleNavigationAviary(scenario="random")
    try:
        env.set_obstacle_count_range(1, 2)
        for seed in range(10):
            env.reset(seed=seed)
            assert 1 <= len(env.obstacles) <= 2
            route = astar_route(
                env.start_position, env.goal, env.obstacles, cruise_z=env.start_position[2]
            )
            assert route is not None, f"seed {seed} generated a scene without an A* route"
    finally:
        env.close()


def test_sensor_policy_history_is_normalized_and_reset_each_episode():
    env = ObstacleNavigationAviary(scenario="wall", action_history=True, max_episode_steps=3)
    try:
        first, _ = env.reset(seed=12)
        assert first.shape == (6 + env.LIDAR_RAYS + 5,)
        assert env.observation_space.contains(first)
        assert np.all(first[-5:] == 0.0)
        next_obs, _, _, _, _ = env.step(np.ones(3, dtype=np.float32))
        assert env.observation_space.contains(next_obs)
        reset_obs, _ = env.reset(seed=13)
        assert np.all(reset_obs[-5:] == 0.0)
    finally:
        env.close()


def test_random_obstacle_scenes_are_reproducible_from_seed():
    env = ObstacleNavigationAviary(scenario="random")
    try:
        env.reset(seed=441)
        first = (env.start_position.copy(), env.goal.copy(), env.obstacles)
        env.reset(seed=441)
        np.testing.assert_array_equal(env.start_position, first[0])
        np.testing.assert_array_equal(env.goal, first[1])
        assert env.obstacles == first[2]
    finally:
        env.close()


def test_configurable_horizontal_speed_limit():
    env = ObstacleNavigationAviary(scenario="wall", max_speed=0.7, max_episode_steps=2)
    try:
        env.reset(seed=10)
        initial_target = env.position_target.copy()
        env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(env.position_target[1] - initial_target[1], 0.7 * env.dt)
        env.set_speed_limit(0.55)
        assert env.max_speed == 0.55
    finally:
        env.close()


def test_speed_limit_rejects_unsafe_or_nonfinite_values():
    env = ObstacleNavigationAviary(scenario="wall")
    for invalid in (0.0, -0.1, 1.21, float("nan"), float("inf")):
        try:
            env.set_speed_limit(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid speed limit {invalid} was accepted")
