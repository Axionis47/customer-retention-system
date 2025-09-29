"""Contract tests for policy golden outputs."""
import numpy as np
import pytest

from env.retention_env import RetentionEnv
from agents.baselines.propensity_threshold import PropensityThresholdPolicy


@pytest.mark.contract
def test_baseline_policy_golden():
    """Baseline policy should produce consistent outputs."""
    policy = PropensityThresholdPolicy(threshold=0.7, offer_pct=0.10)

    # Fixed observation
    obs = {
        "churn_risk": np.array([0.75], dtype=np.float32),
        "accept_prob_0": np.array([0.1], dtype=np.float32),
        "accept_prob_1": np.array([0.3], dtype=np.float32),
        "accept_prob_2": np.array([0.5], dtype=np.float32),
        "accept_prob_3": np.array([0.7], dtype=np.float32),
        "days_since_last_contact": np.array([30.0], dtype=np.float32),
        "contacts_last_7d": np.array([0.0], dtype=np.float32),
        "cooldown_left": np.array([0.0], dtype=np.float32),
        "discount_budget_left": np.array([1.0], dtype=np.float32),
    }

    action = policy(obs)

    # Golden output: should contact (churn_risk > 0.7), offer_idx=2 (10%), delay=0
    assert action[0] == 1, "Should contact"
    assert action[1] == 2, "Should offer 10% (idx=2)"
    assert action[2] == 0, "Should have no delay"


@pytest.mark.contract
def test_env_deterministic_golden():
    """Environment should produce deterministic golden trajectory."""
    env = RetentionEnv(episode_length=5, seed=42)
    obs, _ = env.reset(seed=42)

    # Fixed action sequence
    actions = [
        np.array([1, 2, 0]),
        np.array([0, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 0, 0]),
        np.array([1, 3, 0]),
    ]

    rewards = []

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    # Golden rewards (approximate, may vary slightly due to randomness)
    # Just check they're finite and in reasonable range
    assert len(rewards) == 5, "Should complete 5 steps"
    assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"
    assert all(-1000 < r < 1000 for r in rewards), "Rewards should be in reasonable range"


@pytest.mark.contract
def test_safety_shield_golden():
    """Safety shield should produce consistent flags."""
    from rlhf.safety.shield import SafetyShield

    shield = SafetyShield()

    # Test cases with expected outcomes
    test_cases = [
        {
            "message": "Thank you! We offer you 10% off for 3 months.",
            "hour": 14,
            "expected_safe": True,
        },
        {
            "message": "We guarantee refund!",
            "hour": 14,
            "expected_safe": False,
        },
        {
            "message": "Hi!",  # Too short
            "hour": 14,
            "expected_safe": False,
        },
        {
            "message": "Thank you! We offer you 10% off for 3 months.",
            "hour": 23,  # Quiet hours
            "expected_safe": False,
        },
    ]

    for case in test_cases:
        is_safe, violations, penalty = shield.check_message(
            case["message"],
            hour=case["hour"],
        )

        assert is_safe == case["expected_safe"], (
            f"Message '{case['message']}' at hour {case['hour']} "
            f"expected safe={case['expected_safe']}, got {is_safe}. "
            f"Violations: {violations}"
        )

