"""Lagrangian dual ascent for constraint satisfaction."""
import numpy as np
from typing import Dict, List


class LagrangianMultipliers:
    """
    Manages Lagrangian multipliers for constraint satisfaction via dual ascent.

    Constraints:
        - Budget: total discount cost <= budget
        - Cooldown: no contact within cooldown period
        - Fatigue: contacts_last_7d <= cap

    Dual ascent: λ_t+1 = max(0, λ_t + α * (constraint_violation))
    """

    def __init__(
        self,
        initial_lambda_budget: float = 1.0,
        initial_lambda_cooldown: float = 1.0,
        initial_lambda_fatigue: float = 1.0,
        step_size: float = 0.01,
        min_lambda: float = 0.0,
        max_lambda: float = 100.0,
    ):
        """
        Initialize Lagrangian multipliers.

        Args:
            initial_lambda_budget: Initial budget multiplier
            initial_lambda_cooldown: Initial cooldown multiplier
            initial_lambda_fatigue: Initial fatigue multiplier
            step_size: Dual ascent step size (alpha)
            min_lambda: Minimum multiplier value (non-negative)
            max_lambda: Maximum multiplier value (for stability)
        """
        self.lambda_budget = initial_lambda_budget
        self.lambda_cooldown = initial_lambda_cooldown
        self.lambda_fatigue = initial_lambda_fatigue

        self.step_size = step_size
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # History for monitoring
        self.history: List[Dict[str, float]] = []

    def update(
        self,
        budget_violation: float,
        cooldown_violation: float,
        fatigue_violation: float,
    ) -> Dict[str, float]:
        """
        Update multipliers via dual ascent.

        Args:
            budget_violation: Budget constraint violation (positive = violated)
            cooldown_violation: Cooldown constraint violation count
            fatigue_violation: Fatigue constraint violation count

        Returns:
            Updated multipliers
        """
        # Dual ascent: λ += α * violation
        self.lambda_budget += self.step_size * budget_violation
        self.lambda_cooldown += self.step_size * cooldown_violation
        self.lambda_fatigue += self.step_size * fatigue_violation

        # Clamp to [min_lambda, max_lambda]
        self.lambda_budget = np.clip(self.lambda_budget, self.min_lambda, self.max_lambda)
        self.lambda_cooldown = np.clip(self.lambda_cooldown, self.min_lambda, self.max_lambda)
        self.lambda_fatigue = np.clip(self.lambda_fatigue, self.min_lambda, self.max_lambda)

        # Record history
        self.history.append(
            {
                "lambda_budget": self.lambda_budget,
                "lambda_cooldown": self.lambda_cooldown,
                "lambda_fatigue": self.lambda_fatigue,
                "budget_violation": budget_violation,
                "cooldown_violation": cooldown_violation,
                "fatigue_violation": fatigue_violation,
            }
        )

        return self.get_multipliers()

    def get_multipliers(self) -> Dict[str, float]:
        """Get current multipliers."""
        return {
            "lambda_budget": self.lambda_budget,
            "lambda_cooldown": self.lambda_cooldown,
            "lambda_fatigue": self.lambda_fatigue,
        }

    def reset(self):
        """Reset multipliers to initial values."""
        self.lambda_budget = 1.0
        self.lambda_cooldown = 1.0
        self.lambda_fatigue = 1.0
        self.history = []

    def get_penalty(
        self,
        budget_violation: float,
        cooldown_violation: float,
        fatigue_violation: float,
    ) -> float:
        """
        Compute Lagrangian penalty for current violations.

        Args:
            budget_violation: Budget constraint violation
            cooldown_violation: Cooldown constraint violation count
            fatigue_violation: Fatigue constraint violation count

        Returns:
            Total Lagrangian penalty
        """
        penalty = (
            self.lambda_budget * budget_violation
            + self.lambda_cooldown * cooldown_violation
            + self.lambda_fatigue * fatigue_violation
        )
        return penalty

