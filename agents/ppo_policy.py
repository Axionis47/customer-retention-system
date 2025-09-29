"""PPO policy trainer for retention environment."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from env.retention_env import RetentionEnv
from agents.lagrangian import LagrangianMultipliers


class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim: int, action_dims: List[int], hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for each action dimension
        self.contact_head = nn.Linear(hidden_dim, action_dims[0])
        self.offer_head = nn.Linear(hidden_dim, action_dims[1])
        self.delay_head = nn.Linear(hidden_dim, action_dims[2])

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.shared(obs)

        contact_logits = self.contact_head(features)
        offer_logits = self.offer_head(features)
        delay_logits = self.delay_head(features)

        value = self.value_head(features)

        return contact_logits, offer_logits, delay_logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log prob and value."""
        contact_logits, offer_logits, delay_logits, value = self.forward(obs)

        # Distributions
        contact_dist = torch.distributions.Categorical(logits=contact_logits)
        offer_dist = torch.distributions.Categorical(logits=offer_logits)
        delay_dist = torch.distributions.Categorical(logits=delay_logits)

        if action is None:
            # Sample
            contact_action = contact_dist.sample()
            offer_action = offer_dist.sample()
            delay_action = delay_dist.sample()
            action = torch.stack([contact_action, offer_action, delay_action], dim=-1)
        else:
            contact_action = action[:, 0]
            offer_action = action[:, 1]
            delay_action = action[:, 2]

        # Log probs
        contact_log_prob = contact_dist.log_prob(contact_action)
        offer_log_prob = offer_dist.log_prob(offer_action)
        delay_log_prob = delay_dist.log_prob(delay_action)

        log_prob = contact_log_prob + offer_log_prob + delay_log_prob

        # Entropy
        entropy = contact_dist.entropy() + offer_dist.entropy() + delay_dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


def flatten_obs(obs_dict: Dict) -> np.ndarray:
    """Flatten dict observation to vector."""
    return np.concatenate([v.flatten() for v in obs_dict.values()])


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages)
    returns = advantages + np.array(values)

    return advantages, returns


def train_ppo(config: Dict):
    """Train PPO policy."""
    # Environment
    env = RetentionEnv(
        episode_length=config["env"]["episode_length"],
        initial_budget=config["env"]["initial_budget"],
        cooldown_days=config["env"]["cooldown_days"],
        fatigue_cap=config["env"]["fatigue_cap"],
        lambda_compliance=config["env"]["lambda_compliance"],
        lambda_fatigue=config["env"]["lambda_fatigue"],
        seed=config["seed"],
    )

    # Observation dimension
    obs_sample = env.reset()[0]
    obs_dim = len(flatten_obs(obs_sample))
    action_dims = [2, 4, 3]  # contact, offer, delay

    # Policy network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(obs_dim, action_dims, hidden_dim=config["ppo"]["hidden_dim"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config["ppo"]["learning_rate"])

    # Lagrangian multipliers
    lagrangian = LagrangianMultipliers(
        step_size=config["ppo"]["lagrangian_step_size"],
    )

    # Training loop
    total_steps = 0
    for episode in range(config["ppo"]["num_episodes"]):
        obs, _ = env.reset()
        done = False

        # Rollout
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        while not done:
            obs_flat = flatten_obs(obs)
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            observations.append(obs_flat)
            actions.append(action_np)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)

            obs = next_obs
            total_steps += 1

        # Compute advantages
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=config["ppo"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_batch = torch.FloatTensor(np.array(observations)).to(device)
        action_batch = torch.LongTensor(np.array(actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(device)
        advantages_batch = torch.FloatTensor(advantages).to(device)
        returns_batch = torch.FloatTensor(returns).to(device)

        # PPO update
        for _ in range(config["ppo"]["update_epochs"]):
            _, new_log_probs, entropy, new_values = policy.get_action_and_value(obs_batch, action_batch)

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - config["ppo"]["clip_epsilon"], 1 + config["ppo"]["clip_epsilon"]) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(new_values, returns_batch)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + config["ppo"]["value_coef"] * value_loss
                + config["ppo"]["entropy_coef"] * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["ppo"]["max_grad_norm"])
            optimizer.step()

        # Update Lagrangian (placeholder - would track violations across episodes)
        avg_reward = np.mean(rewards)

        if episode % 10 == 0:
            print(f"Episode {episode}, Steps: {total_steps}, Avg Reward: {avg_reward:.2f}")

    # Save model
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_dir / "ppo_policy.pth")
    print(f"Model saved to {output_dir / 'ppo_policy.pth'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/ppo.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_ppo(config)


if __name__ == "__main__":
    main()

