"""PPO for text generation with adaptive KL."""
import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from rlhf.utils import load_jsonl, format_prompt
from rlhf.safety.shield import SafetyShield


def convert_gcs_path(path: str) -> str:
    """Convert /gcs/ prefix to gs:// for GCS paths."""
    if path.startswith("/gcs/"):
        return "gs://" + path[5:]
    return path


class RewardModel(nn.Module):
    """Reward model with scalar output."""

    def __init__(self, base_model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(pooled)
        return reward


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


def train_ppo_text(config: Dict, sft_path: str = None, rm_path: str = None, output_path: str = None):
    """Train PPO for text generation."""
    print("="*60)
    print("Starting PPO Text Training")
    print("="*60)

    # Override paths if provided
    if sft_path:
        config["sft_path"] = sft_path
    if rm_path:
        config["rm_path"] = rm_path
    if output_path:
        config["output_dir"] = output_path

    print(f"\nConfig: {json.dumps(config, indent=2)}")

    print("\nLoading models...")

    # Get base model name
    base_model = config.get("base_model", "facebook/opt-350m")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model
    sft_model_path = convert_gcs_path(config.get("sft_path", config.get("sft_model_path")))
    print(f"Loading SFT model from {sft_model_path}...")

    if sft_model_path.startswith("gs://"):
        # Download from GCS to local temp
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        local_sft = Path("/tmp/sft_model")
        local_sft.mkdir(parents=True, exist_ok=True)

        # Download all files
        for file in fs.ls(sft_model_path):
            local_file = local_sft / Path(file).name
            with fs.open(file, 'rb') as src:
                with open(local_file, 'wb') as dst:
                    dst.write(src.read())
        sft_model_path = str(local_sft)

    # Policy model (from SFT)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
    print("✓ SFT model loaded")

    # Reference model (frozen SFT)
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
    ref_model.eval()
    print("✓ Reference model loaded")

    # Load trained reward model
    rm_model_path = convert_gcs_path(config.get("rm_path", config.get("rm_model_path")))
    print(f"\nLoading Reward Model from {rm_model_path}...")

    reward_model = None
    if rm_model_path:
        try:
            if rm_model_path.startswith("gs://"):
                # Download from GCS
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                local_rm = Path("/tmp/rm_model")
                local_rm.mkdir(parents=True, exist_ok=True)

                # Download model file
                rm_file = f"{rm_model_path}/reward_model.pth"
                with fs.open(rm_file, 'rb') as src:
                    with open(local_rm / "reward_model.pth", 'wb') as dst:
                        dst.write(src.read())

                # Load model
                reward_model = RewardModel(base_model)
                reward_model.load_state_dict(torch.load(local_rm / "reward_model.pth"))
                reward_model.eval()
                print("✓ Reward model loaded from GCS")
            else:
                rm_file = Path(rm_model_path) / "reward_model.pth"
                if rm_file.exists():
                    reward_model = RewardModel(base_model)
                    reward_model.load_state_dict(torch.load(rm_file))
                    reward_model.eval()
                    print("✓ Reward model loaded")
        except Exception as e:
            print(f"⚠ Failed to load reward model: {e}")
            print("⚠ Will use heuristic rewards")

    if reward_model is None:
        print("⚠ No reward model loaded, using heuristic rewards")

    # Safety shield
    safety_shield = SafetyShield()

    # PPO config
    ppo_config = PPOConfig(
        model_name=base_model,
        learning_rate=config.get("learning_rate", 1.4e-5),
        batch_size=config.get("batch_size", 16),
        mini_batch_size=config.get("mini_batch_size", 4),
        ppo_epochs=config.get("ppo_epochs", 4),
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
        init_beta=config.get("init_beta", 0.1),
        target_kl=config.get("target_kl", 0.15),
    )

    # Load prompts - use validation data for PPO training
    data_path = config.get("data_path", "data/processed/oasst1/sft_valid.jsonl")
    print(f"\nLoading prompts from {data_path}...")
    data = load_jsonl(data_path)
    prompts = [item["prompt"] for item in data[:1000]]  # Limit for budget
    print(f"Loaded {len(prompts)} prompts")

    # Move reward model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if reward_model:
        reward_model = reward_model.to(device)

    print(f"\nUsing device: {device}")
    print(f"Starting PPO training for {config.get('max_steps', 1000)} steps...")

    # Training loop
    step = 0
    max_steps = config.get("max_steps", 1000)

    for prompt in prompts:
        if step >= max_steps:
            break

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate response
        response_ids = ppo_trainer.generate(
            input_ids,
            max_new_tokens=config.get("max_new_tokens", 128),
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

        # 3. Reward model score - USE TRAINED MODEL!
        if reward_model:
            with torch.no_grad():
                # Tokenize full text
                full_text = f"{prompt}\n{response_text}"
                rm_inputs = tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                rm_score = reward_model(rm_inputs["input_ids"], rm_inputs["attention_mask"]).item()
        else:
            # Fallback heuristic
            rm_score = 1.0

        # Total reward
        reward = rm_score - safety_penalty - length_penalty

        # PPO step
        stats = ppo_trainer.step([input_ids[0]], [response_ids[0]], [torch.tensor(reward)])

        # Update KL controller
        if "kl" in stats:
            kl_controller.update(stats["kl"])

        step += 1

        if step % config.get("log_interval", 10) == 0:
            print(
                f"Step {step}/{max_steps}, Reward: {reward:.2f}, "
                f"RM Score: {rm_score:.2f}, Beta: {kl_controller.get_beta():.4f}"
            )

    # Save
    output_dir = convert_gcs_path(config["output_dir"])
    print(f"\nSaving model to {output_dir}...")

    if output_dir.startswith("gs://"):
        # Save locally first then upload
        local_output = Path("/tmp/ppo_text_model")
        local_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_output)
        tokenizer.save_pretrained(local_output)

        # Upload to GCS
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        for file in local_output.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(local_output)
                gcs_path = f"{output_dir}/{rel_path}"
                with open(file, 'rb') as src:
                    with fs.open(gcs_path, 'wb') as dst:
                        dst.write(src.read())
        print(f"✓ Model uploaded to {output_dir}")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO for text generation")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--sft-path", default=None, help="Path to trained SFT model")
    parser.add_argument("--rm-path", default=None, help="Path to trained reward model")
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

    # Extract PPO text config from experiment config
    if "ppo_text" in full_config:
        ppo_config = full_config["ppo_text"]
        # Add global settings
        if "global" in full_config:
            ppo_config.update(full_config["global"])
    else:
        # Assume it's a standalone PPO text config
        ppo_config = full_config

    # Train
    train_ppo_text(ppo_config, args.sft_path, args.rm_path, args.output)

    print("\n" + "="*60)
    print("✓ PPO Text Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

