"""Reward Model training with Bradley-Terry loss."""
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from rlhf.utils import load_jsonl, count_parameters


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
    if data_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path) as f:
            data = [eval(line) for line in f]
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


def train_rm(config: Dict):
    """Train reward model."""
    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(config["model_name"])
    print(f"Trainable parameters: {count_parameters(model)}")

    # Load data
    print(f"Loading data from {config['data_path']}...")
    dataset = load_rm_data(config["data_path"])
    print(f"Loaded {len(dataset)} pairs")

    # Preprocess
    dataset = dataset.map(
        lambda x: preprocess_pairs(x, tokenizer, config["max_length"]),
        batched=True,
    )

    # Custom training loop (simplified)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    model.train()
    for epoch in range(config["num_epochs"]):
        total_loss = 0.0

        for i in range(0, len(dataset), config["batch_size"]):
            batch = dataset[i : i + config["batch_size"]]

            chosen_input_ids = torch.tensor(batch["chosen_input_ids"]).to(device)
            chosen_attention_mask = torch.tensor(batch["chosen_attention_mask"]).to(device)
            rejected_input_ids = torch.tensor(batch["rejected_input_ids"]).to(device)
            rejected_attention_mask = torch.tensor(batch["rejected_attention_mask"]).to(device)

            # Forward
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

            # Loss
            loss = bradley_terry_loss(chosen_rewards, rejected_rewards, margin=config["margin"])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(dataset) // config["batch_size"])
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")

    # Save
    output_path = Path(config["output_dir"]) / "rm_model"
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "reward_model.pth")
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/rm.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_rm(config)


if __name__ == "__main__":
    main()

