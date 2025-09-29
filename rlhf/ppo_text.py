"""PPO for text generation with adaptive KL."""
import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from rlhf.utils import load_jsonl, format_prompt
from rlhf.safety.shield import SafetyShield


def compute_kl_penalty(logprobs, ref_logprobs, beta):
    """Compute KL divergence penalty."""
    kl = logprobs - ref_logprobs
    return -beta * kl.sum(-1)


class AdaptiveKLController:
    """Adaptive KL coefficient controller."""

    def __init__(self, init_beta: float = 0.1, target_kl: float = 0.01, alpha: float = 0.1):
        self.beta = init_beta
        self.target_kl = target_kl
        self.alpha = alpha

    def update(self, current_kl: float):
        """Update beta based on current KL."""
        if current_kl > self.target_kl * 1.5:
            self.beta *= 1.1  # Increase penalty
        elif current_kl < self.target_kl * 0.5:
            self.beta *= 0.9  # Decrease penalty

        self.beta = max(0.01, min(self.beta, 1.0))  # Clamp

    def get_beta(self) -> float:
        return self.beta


def train_ppo_text(config: Dict):
    """Train PPO for text generation."""
    print("Loading models...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (from SFT)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config["sft_model_path"])

    # Reference model (frozen SFT)
    ref_model = AutoModelForCausalLM.from_pretrained(config["sft_model_path"])
    ref_model.eval()

    # Reward model (placeholder - load from rm_train.py output)
    # For now, use simple heuristic reward
    safety_shield = SafetyShield()

    # PPO config
    ppo_config = PPOConfig(
        model_name=config["model_name"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        mini_batch_size=config["mini_batch_size"],
        ppo_epochs=config["ppo_epochs"],
    )

    # PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # Adaptive KL controller
    kl_controller = AdaptiveKLController(
        init_beta=config["init_beta"],
        target_kl=config["target_kl"],
    )

    # Load prompts
    print(f"Loading prompts from {config['data_path']}...")
    data = load_jsonl(config["data_path"])
    prompts = [item["prompt"] for item in data]

    # Training loop
    for epoch in range(config["num_epochs"]):
        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate response
            response_ids = ppo_trainer.generate(
                input_ids,
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Compute reward
            # 1. Safety penalty
            safety_penalty = safety_shield.get_penalty(response_text)

            # 2. Length penalty (prefer concise)
            length_penalty = max(0, len(response_text) - 150) * 0.01

            # 3. Reward model score (placeholder)
            rm_score = 1.0  # Would use trained RM here

            # Total reward
            reward = rm_score - safety_penalty - length_penalty

            # PPO step
            stats = ppo_trainer.step([input_ids[0]], [response_ids[0]], [torch.tensor(reward)])

            # Update KL controller
            if "kl" in stats:
                kl_controller.update(stats["kl"])

            if i % config["logging_steps"] == 0:
                print(
                    f"Epoch {epoch}, Step {i}, Reward: {reward:.2f}, "
                    f"Beta: {kl_controller.get_beta():.4f}"
                )

    # Save
    output_path = Path(config["output_dir"]) / "ppo_text_model"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/ppo_text.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_ppo_text(config)


if __name__ == "__main__":
    main()

