"""Stress tests for retention policy robustness."""
import numpy as np
from typing import Callable, Dict, List

from env.retention_env import RetentionEnv


def stress_test_budget_shift(
    policy: Callable,
    budget_multipliers: List[float] = [0.7, 1.0, 1.3],
    num_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Test policy under different budget constraints.

    Args:
        policy: Policy function (obs -> action)
        budget_multipliers: Budget scaling factors
        num_episodes: Episodes per budget level
        seed: Random seed

    Returns:
        Results dict with rewards per budget level
    """
    results = {"budget_multiplier": [], "avg_reward": [], "violation_rate": []}

    for multiplier in budget_multipliers:
        env = RetentionEnv(
            initial_budget=1000.0 * multiplier,
            seed=seed,
        )

        episode_rewards = []
        episode_violations = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0
            violations = 0

            while not done:
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                violations += info.get("violations", 0)

            episode_rewards.append(total_reward)
            episode_violations.append(violations)

        results["budget_multiplier"].append(multiplier)
        results["avg_reward"].append(np.mean(episode_rewards))
        results["violation_rate"].append(np.mean(episode_violations))

    return results


def stress_test_churn_shift(
    policy: Callable,
    churn_shifts: List[float] = [-0.1, 0.0, 0.1],
    num_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Test policy under shifted churn risk distribution.

    Args:
        policy: Policy function
        churn_shifts: Additive shifts to churn risk
        num_episodes: Episodes per shift
        seed: Random seed

    Returns:
        Results dict
    """
    results = {"churn_shift": [], "avg_reward": []}

    for shift in churn_shifts:
        env = RetentionEnv(seed=seed)

        episode_rewards = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0

            while not done:
                # Apply shift to churn risk
                obs["churn_risk"] = np.clip(obs["churn_risk"] + shift, 0, 1)

                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            episode_rewards.append(total_reward)

        results["churn_shift"].append(shift)
        results["avg_reward"].append(np.mean(episode_rewards))

    return results


def stress_test_accept_shift(
    policy: Callable,
    accept_multipliers: List[float] = [0.8, 1.0, 1.2],
    num_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Test policy under shifted acceptance probabilities.

    Args:
        policy: Policy function
        accept_multipliers: Multiplicative shifts to accept probs
        num_episodes: Episodes per shift
        seed: Random seed

    Returns:
        Results dict
    """
    results = {"accept_multiplier": [], "avg_reward": []}

    for multiplier in accept_multipliers:
        env = RetentionEnv(seed=seed)

        episode_rewards = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0

            while not done:
                # Apply multiplier to accept probs
                for i in range(4):
                    key = f"accept_prob_{i}"
                    obs[key] = np.clip(obs[key] * multiplier, 0, 1)

                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            episode_rewards.append(total_reward)

        results["accept_multiplier"].append(multiplier)
        results["avg_reward"].append(np.mean(episode_rewards))

    return results

