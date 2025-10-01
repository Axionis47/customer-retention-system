"""Retention environment for PPO training with Lagrangian constraints."""
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RetentionEnv(gym.Env):
    """
    Retention environment with budget, cooldown, and fatigue constraints.

    Observation space:
        - churn_risk: [0, 1]
        - accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3: [0, 1] for each offer
        - days_since_last_contact: [0, inf)
        - contacts_last_7d: [0, inf)
        - cooldown_left: [0, inf)
        - discount_budget_left: [0, 1] (normalized)

    Action space:
        - contact: {0, 1}
        - offer_idx: {0, 1, 2, 3} (0%, 5%, 10%, 20%)
        - delay_idx: {0, 1, 2} (0, 1, 3 days)

    Reward:
        reward = revenue_retained - offer_cost - lambda_compliance * violation - lambda_fatigue * fatigue_penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int = 100,
        initial_budget: float = 1000.0,
        cooldown_days: int = 7,
        fatigue_cap: int = 3,
        lambda_compliance: float = 10.0,
        lambda_fatigue: float = 5.0,
        avg_revenue: float = 100.0,
        seed: Optional[int] = None,
        model_path: Optional[str] = None,
        risk_model: Optional[Any] = None,
        accept_model: Optional[Any] = None,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.initial_budget = initial_budget
        self.cooldown_days = cooldown_days
        self.fatigue_cap = fatigue_cap
        self.lambda_compliance = lambda_compliance
        self.lambda_fatigue = lambda_fatigue
        self.avg_revenue = avg_revenue

        # Offer percentages
        self.offers = np.array([0.0, 0.05, 0.10, 0.20])

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "churn_risk": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "accept_prob_0": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "accept_prob_1": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "accept_prob_2": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "accept_prob_3": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "days_since_last_contact": spaces.Box(0, 365, shape=(1,), dtype=np.float32),
                "contacts_last_7d": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
                "cooldown_left": spaces.Box(0, 30, shape=(1,), dtype=np.float32),
                "discount_budget_left": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            }
        )

        # Action space: MultiDiscrete [contact(2), offer_idx(4), delay_idx(3)]
        self.action_space = spaces.MultiDiscrete([2, 4, 3])

        # Load models - prioritize directly passed models
        self.churn_model = risk_model
        self.accept_model = accept_model

        # If not provided, try loading from path
        if (self.churn_model is None or self.accept_model is None) and model_path:
            if Path(model_path).exists():
                self._load_models(model_path)

        # State
        self.np_random = None
        self.seed(seed)
        self._reset_state()

        # Log model status
        if self.churn_model is not None:
            print("✓ RetentionEnv using REAL churn risk model")
        else:
            print("⚠ RetentionEnv using SYNTHETIC churn risk")

        if self.accept_model is not None:
            print("✓ RetentionEnv using REAL acceptance model")
        else:
            print("⚠ RetentionEnv using SYNTHETIC acceptance probabilities")

    def _load_models(self, model_path: str):
        """Load churn and acceptance models."""
        try:
            churn_path = Path(model_path) / "churn_model.pkl"
            accept_path = Path(model_path) / "accept_model.pkl"

            if churn_path.exists():
                with open(churn_path, "rb") as f:
                    self.churn_model = pickle.load(f)["model"]

            if accept_path.exists():
                with open(accept_path, "rb") as f:
                    self.accept_model = pickle.load(f)["model"]
        except Exception as e:
            print(f"Warning: Could not load models: {e}")

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _reset_state(self):
        """Reset internal state."""
        self.step_count = 0
        self.budget_left = self.initial_budget
        self.cooldown_left = 0
        self.days_since_last_contact = 30
        self.contact_history = []  # Last 7 days
        self.violation_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self._reset_state()

        # Generate initial customer
        obs = self._generate_observation()
        info = {"violations": 0, "budget_left": self.budget_left}

        return obs, info

    def _sample_customer_features(self) -> np.ndarray:
        """Sample synthetic customer features for model input."""
        # Generate features matching training data distribution
        # These are rough approximations - in production, use real customer data
        features = {
            'tenure': self.np_random.exponential(24),  # months
            'MonthlyCharges': self.np_random.normal(65, 30),
            'TotalCharges': self.np_random.normal(2000, 2000),
            'Contract_Month-to-month': self.np_random.choice([0, 1], p=[0.4, 0.6]),
            'Contract_One year': self.np_random.choice([0, 1], p=[0.7, 0.3]),
            'Contract_Two year': self.np_random.choice([0, 1], p=[0.8, 0.2]),
            'PaymentMethod_Electronic check': self.np_random.choice([0, 1], p=[0.6, 0.4]),
            'InternetService_Fiber optic': self.np_random.choice([0, 1], p=[0.5, 0.5]),
            'TechSupport_No': self.np_random.choice([0, 1], p=[0.3, 0.7]),
        }

        # Convert to array (order matters - should match training)
        # This is a simplified version - real implementation needs exact feature order
        return np.array(list(features.values())).reshape(1, -1)

    def _generate_observation(self) -> Dict[str, np.ndarray]:
        """Generate observation for current customer."""
        # Sample customer features
        customer_features = self._sample_customer_features()

        # Compute churn risk using REAL model if available
        if self.churn_model is not None:
            try:
                churn_risk = float(self.churn_model.predict_proba(customer_features)[0, 1])
            except Exception as e:
                # Fallback to synthetic if model fails
                print(f"⚠ Churn model prediction failed: {e}, using synthetic")
                churn_risk = self.np_random.beta(2, 5)
        else:
            # Synthetic churn risk
            churn_risk = self.np_random.beta(2, 5)

        # Acceptance probabilities for each offer
        accept_probs = []
        for offer_idx, offer_pct in enumerate(self.offers):
            if self.accept_model is not None:
                try:
                    # Add offer feature to customer features
                    # In real implementation, this should match training data format
                    offer_features = np.concatenate([
                        customer_features,
                        [[offer_pct]]  # Add offer percentage as feature
                    ], axis=1)
                    prob = float(self.accept_model.predict_proba(offer_features)[0, 1])
                except Exception as e:
                    # Fallback to synthetic if model fails
                    logit = -1.0 + 3.0 * offer_pct + 2.0 * churn_risk
                    prob = 1.0 / (1.0 + np.exp(-logit))
            else:
                # Synthetic acceptance probability
                logit = -1.0 + 3.0 * offer_pct + 2.0 * churn_risk
                prob = 1.0 / (1.0 + np.exp(-logit))

            accept_probs.append(prob)

        obs = {
            "churn_risk": np.array([churn_risk], dtype=np.float32),
            "accept_prob_0": np.array([accept_probs[0]], dtype=np.float32),
            "accept_prob_1": np.array([accept_probs[1]], dtype=np.float32),
            "accept_prob_2": np.array([accept_probs[2]], dtype=np.float32),
            "accept_prob_3": np.array([accept_probs[3]], dtype=np.float32),
            "days_since_last_contact": np.array([self.days_since_last_contact], dtype=np.float32),
            "contacts_last_7d": np.array([len(self.contact_history)], dtype=np.float32),
            "cooldown_left": np.array([self.cooldown_left], dtype=np.float32),
            "discount_budget_left": np.array(
                [self.budget_left / self.initial_budget], dtype=np.float32
            ),
        }

        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return next state."""
        contact, offer_idx, delay_idx = action

        # Delays: 0, 1, 3 days
        delay_map = {0: 0, 1: 1, 2: 3}
        delay = delay_map[delay_idx]

        # Update time
        self.step_count += 1
        self.days_since_last_contact += 1
        self.cooldown_left = max(0, self.cooldown_left - 1)

        # Update contact history (sliding window)
        self.contact_history = [c for c in self.contact_history if c > self.step_count - 7]

        reward = 0.0
        violation = 0.0
        info = {}

        if contact == 1:
            # Check cooldown constraint
            if self.cooldown_left > 0:
                violation += 1.0
                self.violation_count += 1
                info["cooldown_violation"] = True

            # Check fatigue constraint
            contacts_last_7d = len(self.contact_history)
            if contacts_last_7d >= self.fatigue_cap:
                fatigue_penalty = contacts_last_7d - self.fatigue_cap + 1
                violation += fatigue_penalty
                info["fatigue_violation"] = True
            else:
                fatigue_penalty = 0

            # Execute contact
            offer_pct = self.offers[offer_idx]
            obs = self._generate_observation()
            churn_risk = obs["churn_risk"][0]
            accept_prob = obs[f"accept_prob_{offer_idx}"][0]

            # Simulate acceptance
            accepted = self.np_random.random() < accept_prob

            if accepted:
                # Revenue retained minus discount cost
                revenue = self.avg_revenue * (1.0 - churn_risk)
                cost = self.avg_revenue * offer_pct
                reward = revenue - cost
                self.budget_left -= cost

                # Budget constraint
                if self.budget_left < 0:
                    violation += abs(self.budget_left) / self.initial_budget
                    info["budget_violation"] = True
            else:
                # No retention
                reward = 0.0

            # Update state
            self.days_since_last_contact = 0
            self.cooldown_left = self.cooldown_days
            self.contact_history.append(self.step_count)

        # Apply Lagrangian penalties
        reward -= self.lambda_compliance * violation
        if contact == 1 and "fatigue_violation" in info:
            reward -= self.lambda_fatigue * fatigue_penalty

        # Terminal condition
        terminated = self.step_count >= self.episode_length
        truncated = False

        # Next observation
        next_obs = self._generate_observation()

        info.update(
            {
                "violations": self.violation_count,
                "budget_left": self.budget_left,
                "step": self.step_count,
            }
        )

        return next_obs, reward, terminated, truncated, info

    def render(self):
        """Render environment (not implemented)."""
        pass

