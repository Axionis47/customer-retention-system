"""Propensity threshold baseline: contact top-p% by churn risk."""
import numpy as np
from typing import Dict, Tuple


class PropensityThresholdPolicy:
    """
    Simple baseline: contact customers with churn_risk > threshold.
    Fixed offer: 10%.
    """

    def __init__(self, threshold: float = 0.7, offer_pct: float = 0.10):
        """
        Initialize propensity threshold policy.

        Args:
            threshold: Churn risk threshold for contact
            offer_pct: Fixed discount percentage (0.10 = 10%)
        """
        self.threshold = threshold
        self.offer_pct = offer_pct

        # Map offer_pct to offer_idx
        offers = [0.0, 0.05, 0.10, 0.20]
        self.offer_idx = min(range(len(offers)), key=lambda i: abs(offers[i] - offer_pct))

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict action based on churn risk.

        Args:
            obs: Observation dict with 'churn_risk'

        Returns:
            action: [contact, offer_idx, delay_idx]
        """
        churn_risk = obs["churn_risk"][0]

        if churn_risk > self.threshold:
            contact = 1
        else:
            contact = 0

        # Fixed offer and no delay
        action = np.array([contact, self.offer_idx, 0])

        return action

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict."""
        return self.predict(obs)

