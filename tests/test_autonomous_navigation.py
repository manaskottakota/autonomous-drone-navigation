"""Contract checks for the project-specific point-to-point environment."""
import numpy as np

from autonomous_navigation.env import PointToPointAviary


def test_reset_observation_and_seeded_scenario():
    env = PointToPointAviary(max_episode_steps=4)
    try:
        obs, info = env.reset(seed=31)
        first_start, first_goal = env.start_position.copy(), env.goal.copy()
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
