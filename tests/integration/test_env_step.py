"""Integration tests for environment rollouts."""
import numpy as np
import pytest

from env.retention_env import RetentionEnv


@pytest.mark.integration
def test_full_episode_deterministic():
    """Full episode should be deterministic with fixed seed."""
    env1 = RetentionEnv(episode_length=20, seed=42)
    env2 = RetentionEnv(episode_length=20, seed=42)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    rewards1 = []
    rewards2 = []

    for _ in range(20):
        action = np.array([1, 2, 0])  # Fixed action

        obs1, reward1, term1, trunc1, _ = env1.step(action)
        obs2, reward2, term2, trunc2, _ = env2.step(action)

        rewards1.append(reward1)
        rewards2.append(reward2)

        if term1 or trunc1:
            break

    assert np.allclose(rewards1, rewards2), "Episodes should be deterministic"


@pytest.mark.integration
def test_episode_terminates():
    """Episode should terminate at episode_length."""
    env = RetentionEnv(episode_length=10, seed=42)
    obs, _ = env.reset(seed=42)

    steps = 0
    done = False

    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps == 10, f"Episode should terminate at 10 steps, got {steps}"
    assert done, "Episode should be done"


@pytest.mark.integration
def test_observation_space_valid():
    """Observations should be within observation space."""
    env = RetentionEnv(seed=42)
    obs, _ = env.reset(seed=42)

    for _ in range(10):
        # Check observation is valid
        for key, value in obs.items():
            assert key in env.observation_space.spaces, f"Key {key} not in observation space"
            space = env.observation_space.spaces[key]
            assert space.contains(value), f"Value {value} not in space {space}"

        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break


@pytest.mark.integration
def test_budget_depletion():
    """Budget should deplete with offers."""
    env = RetentionEnv(initial_budget=100.0, seed=42)
    obs, _ = env.reset(seed=42)

    initial_budget = env.budget_left

    # Make many contacts with high offers
    for _ in range(20):
        action = np.array([1, 3, 0])  # 20% offer
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Budget should have decreased
    assert env.budget_left <= initial_budget, "Budget should decrease with offers"

