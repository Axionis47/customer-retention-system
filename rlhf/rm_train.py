"""Reward Model training with Bradley-Terry loss."""
import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from rlhf.utils import load_jsonl, count_parameters


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
        # Use [CLS] token or mean pooling
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        reward = self.reward_head(pooled)
        return reward


def bradley_terry_loss(chosen_rewards, rejected_rewards, margin=0.0):
    """
    Bradley-Terry pairwise ranking loss.

    Loss = -log(sigmoid(chosen_reward - rejected_reward - margin))
    """
    diff = chosen_rewards - rejected_rewards - margin
    loss = -torch.nn.functional.logsigmoid(diff).mean()
    return loss


def load_rm_data(data_path: str) -> Dataset:
    """Load reward model training data (preference pairs)."""
    data_path = convert_gcs_path(data_path)

    if data_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        data = load_jsonl(data_path)

    dataset = Dataset.from_list(data)
    return dataset


def preprocess_pairs(examples, tokenizer, max_length=256):
    """Preprocess preference pairs."""
    prompts = examples["prompt"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]

    # Tokenize chosen
    chosen_texts = [f"{p}\n{c}" for p, c in zip(prompts, chosen)]
    chosen_inputs = tokenizer(
        chosen_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Tokenize rejected
    rejected_texts = [f"{p}\n{r}" for p, r in zip(prompts, rejected)]
    rejected_inputs = tokenizer(
        rejected_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
    }


def train_rm(config: Dict, train_data_path: str = None, valid_data_path: str = None, output_path: str = None):
    """Train reward model."""
    print("="*60)
    print("Starting Reward Model Training")
    print("="*60)

    # Override paths if provided
    if train_data_path:
        config["train_data_path"] = train_data_path
    if valid_data_path:
        config["valid_data_path"] = valid_data_path
    if output_path:
        config["output_dir"] = output_path

    print(f"Config: {json.dumps(config, indent=2)}")

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(config["base_model"])
    print(f"Trainable parameters: {count_parameters(model)}")

    # Load data
    train_path = config.get("train_data_path", config.get("data_path"))
    print(f"\nLoading training data from {train_path}...")
    dataset = load_rm_data(train_path)
    print(f"Loaded {len(dataset)} training pairs")

    # Load validation data if available
    valid_dataset = None
    if "valid_data_path" in config and config["valid_data_path"]:
        print(f"Loading validation data from {config['valid_data_path']}...")
        valid_dataset = load_rm_data(config["valid_data_path"])
        print(f"Loaded {len(valid_dataset)} validation pairs")

    # Preprocess
    max_length = config.get("max_length", 512)
    print(f"\nPreprocessing with max_length={max_length}...")
    dataset = dataset.map(
        lambda x: preprocess_pairs(x, tokenizer, max_length),
        batched=True,
    )

    if valid_dataset:
        valid_dataset = valid_dataset.map(
            lambda x: preprocess_pairs(x, tokenizer, max_length),
            batched=True,
        )

    # Custom training loop (simplified)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-5))

    max_steps = config.get("max_steps", 1000)
    batch_size = config.get("batch_size", 4)
    margin = config.get("margin", 0.0)

    print(f"\nTraining for {max_steps} steps with batch_size={batch_size}...")

    model.train()
    step = 0
    total_loss = 0.0

    while step < max_steps:
        for i in range(0, len(dataset), batch_size):
            if step >= max_steps:
                break

            batch = dataset[i : i + batch_size]

            chosen_input_ids = torch.tensor(batch["chosen_input_ids"]).to(device)
            chosen_attention_mask = torch.tensor(batch["chosen_attention_mask"]).to(device)
            rejected_input_ids = torch.tensor(batch["rejected_input_ids"]).to(device)
            rejected_attention_mask = torch.tensor(batch["rejected_attention_mask"]).to(device)

            # Forward
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

            # Loss
            loss = bradley_terry_loss(chosen_rewards, rejected_rewards, margin=margin)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % config.get("log_interval", 10) == 0:
                avg_loss = total_loss / step
                print(f"Step {step}/{max_steps}, Loss: {avg_loss:.4f}")

    # Save
    output_dir = convert_gcs_path(config["output_dir"])
    print(f"\nSaving model to {output_dir}...")

    if output_dir.startswith("gs://"):
        # For GCS, save locally first then upload
        local_output = Path("/tmp/rm_model")
        local_output.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), local_output / "reward_model.pth")
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
        torch.save(model.state_dict(), output_path / "reward_model.pth")
        tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model with Bradley-Terry loss")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--train-data", default=None, help="Override training data path")
    parser.add_argument("--valid-data", default=None, help="Override validation data path")
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

    # Extract RM config from experiment config
    if "reward_model" in full_config:
        rm_config = full_config["reward_model"]
        # Add global settings
        if "global" in full_config:
            rm_config.update(full_config["global"])
    else:
        # Assume it's a standalone RM config
        rm_config = full_config

    # Train
    train_rm(rm_config, args.train_data, args.valid_data, args.output)

    print("\n" + "="*60)
    print("✓ Reward Model Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

