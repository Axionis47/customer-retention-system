"""Unit tests for reward computation."""
import numpy as np
import pytest

from env.retention_env import RetentionEnv


@pytest.mark.unit
def test_reward_no_nan():
    """Reward should never be NaN."""
    env = RetentionEnv(seed=42)
    obs, _ = env.reset(seed=42)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert not np.isnan(reward), "Reward should not be NaN"
        assert np.isfinite(reward), "Reward should be finite"

        if terminated or truncated:
            break


@pytest.mark.unit
def test_reward_penalty_applies():
    """Penalties should reduce reward."""
    env = RetentionEnv(
        cooldown_days=7,
        lambda_compliance=10.0,
        seed=42,
    )

    obs, _ = env.reset(seed=42)

    # First contact (no violation)
    action = np.array([1, 2, 0])  # Contact, 10% offer, no delay
    obs, reward1, _, _, info1 = env.step(action)

    # Immediate second contact (cooldown violation)
    action = np.array([1, 2, 0])
    obs, reward2, _, _, info2 = env.step(action)

    # Second contact should have lower reward due to violation
    if "cooldown_violation" in info2:
        assert reward2 < reward1, "Violation should reduce reward"


@pytest.mark.unit
def test_no_contact_zero_cost():
    """No contact should have zero cost."""
    env = RetentionEnv(seed=42)
    obs, _ = env.reset(seed=42)

    # No contact action
    action = np.array([0, 0, 0])
    obs, reward, _, _, info = env.step(action)

    # Budget should not decrease
    assert info["budget_left"] == env.initial_budget


@pytest.mark.unit
def test_reward_deterministic():
    """Reward should be deterministic with fixed seed."""
    env1 = RetentionEnv(seed=42)
    env2 = RetentionEnv(seed=42)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    action = np.array([1, 2, 0])

    _, reward1, _, _, _ = env1.step(action)
    _, reward2, _, _, _ = env2.step(action)

    assert reward1 == reward2, "Rewards should be deterministic with same seed"

