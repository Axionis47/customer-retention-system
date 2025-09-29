"""Thompson Sampling bandit for offer selection."""
import numpy as np
from typing import Dict


class ThompsonSamplingBandit:
    """
    Thompson Sampling over discrete offers {0%, 5%, 10%, 20%}.
    Beta prior for each arm.
    """

    def __init__(self, num_arms: int = 4, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson Sampling bandit.

        Args:
            num_arms: Number of offer options
            alpha_prior: Beta distribution alpha prior
            beta_prior: Beta distribution beta prior
        """
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms) * alpha_prior
        self.beta = np.ones(num_arms) * beta_prior

    def select_arm(self) -> int:
        """Sample from posterior and select arm with highest sample."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """
        Update posterior based on observed reward.

        Args:
            arm: Selected arm index
            reward: Observed reward (1 = success, 0 = failure)
        """
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict action using Thompson Sampling.

        Args:
            obs: Observation dict

        Returns:
            action: [contact, offer_idx, delay_idx]
        """
        churn_risk = obs["churn_risk"][0]

        # Contact if high churn risk
        if churn_risk > 0.6:
            contact = 1
            offer_idx = self.select_arm()
        else:
            contact = 0
            offer_idx = 0

        action = np.array([contact, offer_idx, 0])

        return action

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict."""
        return self.predict(obs)

