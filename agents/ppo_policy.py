"""PPO policy trainer for retention environment."""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from env.retention_env import RetentionEnv
from agents.lagrangian import LagrangianMultipliers


def convert_gcs_path(path: str) -> str:
    """Convert /gcs/ prefix to gs:// for GCS paths."""
    if path.startswith("/gcs/"):
        return "gs://" + path[5:]
    return path


def load_model_from_path(model_path: str):
    """Load a pickled model from local or GCS path."""
    model_path = convert_gcs_path(model_path)

    if model_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(model_path, 'rb') as f:
            artifact = pickle.load(f)
    else:
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)

    return artifact["model"]


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


def train_ppo(config: Dict, risk_model_path: str = None, accept_model_path: str = None, output_path: str = None):
    """Train PPO policy."""
    print("="*60)
    print("Starting PPO Decision Policy Training")
    print("="*60)

    # Override paths if provided
    if risk_model_path:
        config["risk_model_path"] = risk_model_path
    if accept_model_path:
        config["accept_model_path"] = accept_model_path
    if output_path:
        config["output_dir"] = output_path

    # Load trained risk and acceptance models
    print("\nLoading trained models...")
    if "risk_model_path" in config:
        print(f"Loading risk model from {config['risk_model_path']}...")
        risk_model = load_model_from_path(config["risk_model_path"])
        print("✓ Risk model loaded")
    else:
        print("⚠ No risk model path provided, using dummy model")
        risk_model = None

    if "accept_model_path" in config:
        print(f"Loading acceptance model from {config['accept_model_path']}...")
        accept_model = load_model_from_path(config["accept_model_path"])
        print("✓ Acceptance model loaded")
    else:
        print("⚠ No acceptance model path provided, using dummy model")
        accept_model = None

    # Environment - check if nested or flat config
    if "environment" in config:
        env_config = config["environment"]
    else:
        # Flat config - use defaults
        env_config = {}

    print(f"\nCreating environment with trained models...")
    env = RetentionEnv(
        episode_length=env_config.get("episode_length", 30),
        initial_budget=env_config.get("initial_budget", 1000.0),
        cooldown_days=env_config.get("cooldown_days", 7),
        fatigue_cap=env_config.get("fatigue_cap", 3),
        lambda_compliance=env_config.get("lambda_compliance", 1.0),
        lambda_fatigue=env_config.get("lambda_fatigue", 1.0),
        seed=config.get("seed", 42),
        risk_model=risk_model,  # ✅ PASS TRAINED MODEL!
        accept_model=accept_model,  # ✅ PASS TRAINED MODEL!
    )

    # Observation dimension
    obs_sample = env.reset()[0]
    obs_dim = len(flatten_obs(obs_sample))
    action_dims = [2, 4, 3]  # contact, offer, delay

    # Policy network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Handle both nested (ppo: {...}) and flat config structures
    if "ppo" in config:
        ppo_config = config["ppo"]
    else:
        # Flat config - use the config directly
        ppo_config = config

    policy = PolicyNetwork(
        obs_dim,
        action_dims,
        hidden_dim=ppo_config.get("hidden_dim", 128)
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=ppo_config.get("learning_rate", 3e-4))

    # Lagrangian multipliers (for future constraint enforcement)
    lagrangian = LagrangianMultipliers(
        step_size=ppo_config.get("lagrangian_step_size", 0.01),
    )

    # Training loop - calculate episodes from total_iters if available
    if "total_iters" in ppo_config:
        # Experiment config style: total_iters * steps_per_iter / episode_length
        steps_per_iter = ppo_config.get("steps_per_iter", 2048)
        episode_length = env_config.get("episode_length", 30)
        total_steps = ppo_config["total_iters"] * steps_per_iter
        num_episodes = max(1, total_steps // episode_length)
    else:
        num_episodes = ppo_config.get("num_episodes", 100)

    print(f"\nTraining for {num_episodes} episodes...")

    total_steps = 0
    for episode in range(num_episodes):
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
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
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
        for _ in range(ppo_config.get("update_epochs", 4)):
            _, new_log_probs, entropy, new_values = policy.get_action_and_value(obs_batch, action_batch)

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(
                ratio,
                1 - ppo_config.get("clip_epsilon", 0.2),
                1 + ppo_config.get("clip_epsilon", 0.2)
            ) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(new_values, returns_batch)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + ppo_config.get("value_coef", 0.5) * value_loss
                + ppo_config.get("entropy_coef", 0.01) * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.get("max_grad_norm", 0.5))
            optimizer.step()

        # Update Lagrangian (placeholder - would track violations across episodes)
        avg_reward = np.mean(rewards)

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Steps: {total_steps}, Avg Reward: {avg_reward:.2f}")

    # Save model
    output_dir = convert_gcs_path(config["output_dir"])
    print(f"\nSaving model to {output_dir}...")

    if output_dir.startswith("gs://"):
        # For GCS, save locally first then upload
        local_output = Path("/tmp/ppo_policy")
        local_output.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), local_output / "ppo_policy.pth")

        # Upload to GCS
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        gcs_path = f"{output_dir}/ppo_policy.pth"
        with open(local_output / "ppo_policy.pth", 'rb') as src:
            with fs.open(gcs_path, 'wb') as dst:
                dst.write(src.read())
        print(f"✓ Model uploaded to {gcs_path}")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), output_path / "ppo_policy.pth")
        print(f"✓ Model saved to {output_path / 'ppo_policy.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO decision policy")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--risk-model", default=None, help="Path to trained risk model")
    parser.add_argument("--accept-model", default=None, help="Path to trained acceptance model")
    parser.add_argument("--output", default=None, help="Override output directory")
    args = parser.parse_args()

    # Load experiment config
    config_path = convert_gcs_path(args.config)
    print(f"Loading config from {config_path}...")

    if config_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

    # Extract PPO decision config from experiment config
    if "ppo_decision" in full_config:
        ppo_config = full_config["ppo_decision"]
        # Add global settings
        if "global" in full_config:
            ppo_config.update(full_config["global"])
    else:
        # Assume it's a standalone PPO config
        ppo_config = full_config

    # Train
    train_ppo(ppo_config, args.risk_model, args.accept_model, args.output)

    print("\n" + "="*60)
    print("✓ PPO Decision Policy Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

