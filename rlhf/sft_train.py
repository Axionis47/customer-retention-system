"""Supervised Fine-Tuning (SFT) with QLoRA."""
import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from rlhf.utils import load_jsonl, count_parameters


def load_sft_data(data_path: str) -> Dataset:
    """Load SFT training data."""
    if data_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path) as f:
            data = [eval(line) for line in f]
    else:
        data = load_jsonl(data_path)

    # Convert to HF dataset
    dataset = Dataset.from_list(data)
    return dataset


def preprocess_function(examples, tokenizer, max_length=256):
    """Preprocess examples for SFT."""
    prompts = examples["prompt"]
    responses = examples["response"]

    # Concatenate prompt + response
    texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def train_sft(config: Dict):
    """Train SFT model with QLoRA."""
    print("Loading tokenizer and model...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (quantized for QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        load_in_8bit=config.get("load_in_8bit", False),  # Set to True for real QLoRA
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training
    if config.get("load_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {count_parameters(model)}")

    # Load data
    print(f"Loading data from {config['data_path']}...")
    dataset = load_sft_data(config["data_path"])
    print(f"Loaded {len(dataset)} examples")

    # Preprocess
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    output_path = Path(config["output_dir"]) / "sft_model"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/sft.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_sft(config)


if __name__ == "__main__":
    main()

