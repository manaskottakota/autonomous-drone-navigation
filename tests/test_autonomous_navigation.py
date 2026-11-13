"""Contract checks for the project-specific point-to-point environment."""
import numpy as np

from autonomous_navigation.env import PointToPointAviary
from autonomous_navigation.env import sample_scenario
from autonomous_navigation.obstacle_env import (
    BoxObstacle, CylinderObstacle, GateObstacle, ObstacleNavigationAviary, SphereObstacle,
)
from autonomous_navigation.obstacle_env import WALL_SCENE
from autonomous_navigation.planning import astar_route
from autonomous_navigation.train_obstacle import transfer_policy_state
import torch


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
    env.close()


def test_obstacle_clearance_margin_grows_with_flight_speed():
    env = ObstacleNavigationAviary(scenario="wall")
    try:
        slow_penalty = env._speed_adaptive_clearance_penalty(clearance_m=0.5, horizontal_speed_mps=0.4)
        fast_penalty = env._speed_adaptive_clearance_penalty(clearance_m=0.5, horizontal_speed_mps=0.9)
        assert slow_penalty == 0.0
        assert fast_penalty > slow_penalty
    finally:
        env.close()


def test_dynamic_obstacle_moves_and_sensor_history_is_bounded():
    env = ObstacleNavigationAviary(
        scenario="dynamic_wall", dynamic_observation=True, max_episode_steps=3,
    )
    try:
        obs, _ = env.reset(seed=17)
        assert obs.shape == (6 + env.LIDAR_RAYS * 2,)
        assert env.observation_space.contains(obs)
        initial_center = env.dynamic_centers[0].copy()
        next_obs, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert env.dynamic_centers[0][1] > initial_center[1]
        assert info["dynamic_obstacle_count"] == 1
        assert env.observation_space.contains(next_obs)
        assert np.all(np.isfinite(next_obs))
    finally:
        env.close()


def test_dynamic_obstacle_reflects_at_bounds_and_swept_collision_uses_motion():
    env = ObstacleNavigationAviary(scenario="dynamic_wall")
    try:
        env.reset(seed=3)
        mover = env.dynamic_obstacles[0]
        env.dynamic_centers[0] = np.asarray(mover.bounds[1], dtype=float) - np.array([0.0, 0.001, 0.0])
        env.dynamic_velocities[0] = np.array([0.0, 0.22, 0.0])
        env._advance_dynamic_obstacles()
        assert env.dynamic_velocities[0][1] < 0

        # A stationary drone is hit by a box that crosses between the two
        # centers, even if neither endpoint overlaps its collision volume.
        env.dynamic_previous_centers[0] = np.array([0.0, -0.5, 0.7])
        env.dynamic_centers[0] = np.array([0.0, 0.5, 0.7])
        assert env._swept_collision(np.array([0.0, 0.0, 0.7]), np.array([0.0, 0.0, 0.7]))
    finally:
        env.close()


def test_policy_frequency_keeps_pid_rate_and_scales_policy_dt():
    env = ObstacleNavigationAviary(scenario="dynamic_wall", policy_hz=24)
    try:
        env.reset(seed=4)
        assert env.policy_hz == 24
        assert env.control_hz == 48
        assert env.action_repeat == 2
        assert env.dt == 1 / 24
        assert env.max_episode_steps == 400
        assert np.isclose(env.max_episode_steps * env.dt, 200 / 12)
    finally:
        env.close()


def test_dynamic_sensor_history_preserves_legacy_policy_feature_prefix():
    env = ObstacleNavigationAviary(
        scenario="dynamic_wall", action_history=True, dynamic_observation=True,
    )
    try:
        obs, _ = env.reset(seed=8)
        assert obs.shape == (6 + env.LIDAR_RAYS + 5 + env.LIDAR_RAYS,)
        np.testing.assert_array_equal(obs[env.base_observation_size:env.base_observation_size + 5], 0.0)
        np.testing.assert_array_equal(obs[-env.LIDAR_RAYS:], 0.0)
    finally:
        env.close()


def test_policy_transfer_copies_legacy_inputs_and_zeroes_new_sensor_inputs():
    source = {
        "mlp_extractor.policy_net.0.weight": torch.full((4, 83), 2.0),
        "action_net.bias": torch.tensor([1.0, 2.0, 3.0]),
    }
    target = {
        "mlp_extractor.policy_net.0.weight": torch.ones((4, 155)),
        "action_net.bias": torch.zeros(3),
    }
    transferred = transfer_policy_state(target, source)
    assert torch.all(transferred["mlp_extractor.policy_net.0.weight"][:, :83] == 2.0)
    assert torch.count_nonzero(transferred["mlp_extractor.policy_net.0.weight"][:, 83:]) == 0
    assert torch.equal(transferred["action_net.bias"], source["action_net.bias"])


def test_dynamic_random_samples_crossing_head_on_and_seeded_velocities():
    env = ObstacleNavigationAviary(scenario="dynamic_random", obstacle_count_range=(1, 1))
    try:
        alignments = []
        for seed in range(40):
            env.reset(seed=seed)
            mover = env.dynamic_obstacles[0]
            direction = env.goal[:2] - env.start_position[:2]
            direction /= np.linalg.norm(direction)
            velocity = np.asarray(mover.velocity[:2])
            speed = float(np.linalg.norm(velocity))
            assert 0.12 <= speed <= 0.30
            alignments.append(float(np.dot(velocity / speed, direction)))
        assert any(abs(value) < 0.3 for value in alignments), "crossing trajectories should be sampled"
        assert any(value > 0.7 for value in alignments), "same-direction trajectories should be sampled"
        assert any(value < -0.7 for value in alignments), "head-on trajectories should be sampled"

        env.set_dynamic_speed_range(0.25, 0.35)
        env.reset(seed=101)
        first = env.dynamic_obstacles
        env.reset(seed=101)
        assert env.dynamic_obstacles == first
        assert 0.25 <= np.linalg.norm(env.dynamic_obstacles[0].velocity[:2]) <= 0.35
    finally:
        env.close()


def test_mixed_obstacle_geometries_create_lidar_hits_and_gate_opening():
    env = ObstacleNavigationAviary(scenario="complex", max_episode_steps=2)
    try:
        obstacles = (
            CylinderObstacle((0.0, -0.7, 0.7), 0.22, 1.2),
            SphereObstacle((0.0, -0.35, 0.7), 0.24),
            GateObstacle((0.0, 0.7, 0.9), 0.9, 1.8),
        )
        obs, _ = env.reset(seed=21, options={
            "start": [-1.1, 0.0, 0.7], "goal": [1.1, 0.0, 0.7], "obstacles": obstacles,
        })
        assert env.observation_space.contains(obs)
        # Cylinder/sphere surfaces and gate bars are actual collision shapes
        # used by the same PyBullet batched ray sensor as policy observations.
        assert len(env.obstacle_ids) == 5
        assert np.any(obs[6:] < 0.95)
        # Gate has a clear center aperture; direct ray at y=0 does not hit its lintel/posts.
        assert env.last_ranges[env.AZIMUTH_RAYS] > 0.95
    finally:
        env.close()


def test_complex_scenes_are_seeded_nonoverlapping_and_reachable():
    env = ObstacleNavigationAviary(scenario="complex")
    try:
        for seed in (4, 12, 29, 61):
            env.reset(seed=seed)
            assert 5 <= len(env.obstacles) <= 10
            sampled = env.obstacles
            assert {type(obstacle) for obstacle in env.obstacles} <= {
                BoxObstacle, CylinderObstacle, SphereObstacle, GateObstacle,
            }
            route = astar_route(env.start_position, env.goal, env.obstacles,
                                cruise_z=env.start_position[2])
            assert route is not None, f"complex scene seed {seed} is not reachable"
            env.reset(seed=seed)
            assert env.obstacles == sampled
    finally:
        env.close()


def test_cylinder_and_sphere_contact_are_detected_by_swept_collision():
    env = ObstacleNavigationAviary(scenario="complex")
    try:
        for obstacle in (CylinderObstacle((0, 0, 0.7), 0.2, 1.0),
                         SphereObstacle((0, 0, 0.7), 0.2)):
            env.obstacles = (obstacle,)
            assert env._swept_collision(np.array([-0.4, 0.0, 0.7]),
                                        np.array([0.4, 0.0, 0.7]))
        env.obstacles = (SphereObstacle((0, 0, 0.7), 0.2),)
        # This diagonal near-miss falls inside the sphere's AABB but outside
        # its round collision envelope; primitive-aware sweeps must not report it.
        assert not env._swept_collision(np.array([-0.4, 0.21, 0.91]),
                                        np.array([0.4, 0.21, 0.91]))
    finally:
        env.close()
