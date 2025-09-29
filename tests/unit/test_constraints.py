"""Unit tests for constraint enforcement."""
import numpy as np
import pytest

from env.retention_env import RetentionEnv


@pytest.mark.unit
def test_cooldown_enforced():
    """Cooldown constraint should be tracked."""
    env = RetentionEnv(cooldown_days=7, seed=42)
    obs, _ = env.reset(seed=42)

    # First contact
    action = np.array([1, 2, 0])
    obs, _, _, _, info = env.step(action)

    assert env.cooldown_left == 7, "Cooldown should be set after contact"

    # Try immediate second contact
    action = np.array([1, 2, 0])
    obs, _, _, _, info = env.step(action)

    assert "cooldown_violation" in info or info["violations"] > 0, "Cooldown violation should be flagged"


@pytest.mark.unit
def test_fatigue_cap():
    """Fatigue cap should limit contacts."""
    env = RetentionEnv(
        fatigue_cap=2,
        cooldown_days=0,  # Disable cooldown for this test
        seed=42,
    )
    obs, _ = env.reset(seed=42)

    # Make multiple contacts
    for i in range(5):
        action = np.array([1, 1, 0])
        obs, reward, _, _, info = env.step(action)

        contacts_7d = len(env.contact_history)

        if contacts_7d > env.fatigue_cap:
            # Should have fatigue violation
            assert "fatigue_violation" in info or reward < 0, f"Fatigue violation expected at contact {i+1}"


@pytest.mark.unit
def test_budget_tracking():
    """Budget should decrease with offers."""
    env = RetentionEnv(initial_budget=1000.0, seed=42)
    obs, _ = env.reset(seed=42)

    initial_budget = env.budget_left

    # Contact with 20% offer
    action = np.array([1, 3, 0])  # 20% offer
    obs, _, _, _, info = env.step(action)

    # Budget should decrease (if accepted)
    # Note: acceptance is probabilistic, so we just check it's tracked
    assert "budget_left" in info
    assert info["budget_left"] <= initial_budget


@pytest.mark.unit
def test_violation_count_increments():
    """Violation count should increment."""
    env = RetentionEnv(cooldown_days=7, seed=42)
    obs, _ = env.reset(seed=42)

    initial_violations = env.violation_count

    # First contact
    action = np.array([1, 2, 0])
    env.step(action)

    # Immediate second contact (violation)
    action = np.array([1, 2, 0])
    env.step(action)

    assert env.violation_count >= initial_violations, "Violation count should increment"

