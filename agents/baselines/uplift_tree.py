"""Uplift tree baseline using two-model approach."""
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier


class UpliftTreePolicy:
    """
    Two-model uplift approach:
    - Model 1: P(retain | contact)
    - Model 2: P(retain | no contact)
    - Uplift = Model1 - Model2
    Contact if uplift > threshold.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        offer_grid: Optional[list] = None,
        seed: int = 42,
    ):
        """
        Initialize uplift tree policy.

        Args:
            threshold: Uplift threshold for contact
            offer_grid: List of offer percentages to try
            seed: Random seed
        """
        self.threshold = threshold
        self.offer_grid = offer_grid or [0.0, 0.05, 0.10, 0.20]
        self.seed = seed

        # Placeholder models (would be trained on historical data)
        self.model_contact = RandomForestClassifier(n_estimators=10, random_state=seed)
        self.model_no_contact = RandomForestClassifier(n_estimators=10, random_state=seed)

        # Fit on dummy data
        X_dummy = np.random.randn(100, 4)
        y_dummy = np.random.randint(0, 2, 100)
        self.model_contact.fit(X_dummy, y_dummy)
        self.model_no_contact.fit(X_dummy, y_dummy)

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict action based on uplift.

        Args:
            obs: Observation dict

        Returns:
            action: [contact, offer_idx, delay_idx]
        """
        # Extract features
        churn_risk = obs["churn_risk"][0]
        days_since = obs["days_since_last_contact"][0]
        contacts_7d = obs["contacts_last_7d"][0]
        budget_left = obs["discount_budget_left"][0]

        features = np.array([[churn_risk, days_since, contacts_7d, budget_left]])

        # Predict uplift
        p_contact = self.model_contact.predict_proba(features)[0, 1]
        p_no_contact = self.model_no_contact.predict_proba(features)[0, 1]
        uplift = p_contact - p_no_contact

        if uplift > self.threshold:
            contact = 1
            # Choose offer with highest expected value (simplified)
            offer_idx = 2  # 10%
        else:
            contact = 0
            offer_idx = 0

        action = np.array([contact, offer_idx, 0])

        return action

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict."""
        return self.predict(obs)

