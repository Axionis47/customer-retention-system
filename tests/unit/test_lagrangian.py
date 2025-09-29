"""Unit tests for Lagrangian multipliers."""
import numpy as np
import pytest

from agents.lagrangian import LagrangianMultipliers


@pytest.mark.unit
def test_multipliers_non_negative():
    """Multipliers should always be non-negative."""
    lagrangian = LagrangianMultipliers(
        initial_lambda_budget=1.0,
        step_size=0.1,
        min_lambda=0.0,
    )

    # Update with negative violations (should not go below 0)
    for _ in range(20):
        lagrangian.update(
            budget_violation=-1.0,
            cooldown_violation=-1.0,
            fatigue_violation=-1.0,
        )

    multipliers = lagrangian.get_multipliers()

    assert multipliers["lambda_budget"] >= 0, "Budget multiplier should be non-negative"
    assert multipliers["lambda_cooldown"] >= 0, "Cooldown multiplier should be non-negative"
    assert multipliers["lambda_fatigue"] >= 0, "Fatigue multiplier should be non-negative"


@pytest.mark.unit
def test_multipliers_increase_with_violations():
    """Multipliers should increase with positive violations."""
    lagrangian = LagrangianMultipliers(
        initial_lambda_budget=1.0,
        step_size=0.1,
    )

    initial = lagrangian.get_multipliers()

    # Update with positive violations
    lagrangian.update(
        budget_violation=10.0,
        cooldown_violation=5.0,
        fatigue_violation=3.0,
    )

    updated = lagrangian.get_multipliers()

    assert updated["lambda_budget"] > initial["lambda_budget"], "Budget multiplier should increase"
    assert updated["lambda_cooldown"] > initial["lambda_cooldown"], "Cooldown multiplier should increase"
    assert updated["lambda_fatigue"] > initial["lambda_fatigue"], "Fatigue multiplier should increase"


@pytest.mark.unit
def test_multipliers_clamped():
    """Multipliers should be clamped to max value."""
    lagrangian = LagrangianMultipliers(
        initial_lambda_budget=1.0,
        step_size=10.0,
        max_lambda=50.0,
    )

    # Large violations
    for _ in range(10):
        lagrangian.update(
            budget_violation=100.0,
            cooldown_violation=100.0,
            fatigue_violation=100.0,
        )

    multipliers = lagrangian.get_multipliers()

    assert multipliers["lambda_budget"] <= 50.0, "Budget multiplier should be clamped"
    assert multipliers["lambda_cooldown"] <= 50.0, "Cooldown multiplier should be clamped"
    assert multipliers["lambda_fatigue"] <= 50.0, "Fatigue multiplier should be clamped"


@pytest.mark.unit
def test_penalty_computation():
    """Penalty should be computed correctly."""
    lagrangian = LagrangianMultipliers(
        initial_lambda_budget=2.0,
        initial_lambda_cooldown=3.0,
        initial_lambda_fatigue=1.0,
    )

    penalty = lagrangian.get_penalty(
        budget_violation=5.0,
        cooldown_violation=2.0,
        fatigue_violation=1.0,
    )

    expected = 2.0 * 5.0 + 3.0 * 2.0 + 1.0 * 1.0
    assert np.isclose(penalty, expected), f"Penalty should be {expected}, got {penalty}"


@pytest.mark.unit
def test_history_tracking():
    """History should be tracked."""
    lagrangian = LagrangianMultipliers()

    assert len(lagrangian.history) == 0, "History should be empty initially"

    lagrangian.update(1.0, 1.0, 1.0)
    lagrangian.update(2.0, 2.0, 2.0)

    assert len(lagrangian.history) == 2, "History should have 2 entries"
    assert "lambda_budget" in lagrangian.history[0]
    assert "budget_violation" in lagrangian.history[0]

