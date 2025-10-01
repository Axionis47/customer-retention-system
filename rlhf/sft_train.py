"""Supervised Fine-Tuning (SFT) with QLoRA."""
import argparse
import json
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


def convert_gcs_path(path: str) -> str:
    """Convert /gcs/ prefix to gs:// for GCS paths."""
    if path.startswith("/gcs/"):
        return "gs://" + path[5:]
    return path


def load_sft_data(data_path: str) -> Dataset:
    """Load SFT training data from local or GCS."""
    data_path = convert_gcs_path(data_path)

    if data_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
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


def train_sft(config: Dict, train_data_path: str = None, valid_data_path: str = None, output_path: str = None):
    """Train SFT model with QLoRA."""
    print("="*60)
    print("Starting SFT Training")
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (quantized for QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        load_in_8bit=config.get("load_in_8bit", False),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training
    if config.get("load_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"].get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {count_parameters(model)}")

    # Load data
    train_path = config.get("train_data_path", config.get("data_path"))
    print(f"\nLoading training data from {train_path}...")
    dataset = load_sft_data(train_path)
    print(f"Loaded {len(dataset)} training examples")

    # Load validation data if available
    valid_dataset = None
    if "valid_data_path" in config and config["valid_data_path"]:
        print(f"Loading validation data from {config['valid_data_path']}...")
        valid_dataset = load_sft_data(config["valid_data_path"])
        print(f"Loaded {len(valid_dataset)} validation examples")

    # Preprocess
    max_length = config.get("max_length", 512)
    print(f"\nPreprocessing with max_length={max_length}...")
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    if valid_dataset:
        valid_dataset = valid_dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=valid_dataset.column_names,
        )

    # Training arguments
    output_dir = convert_gcs_path(config["output_dir"])
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=config.get("max_steps", 1000),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_steps=config.get("warmup_steps", 100),
        logging_steps=config.get("log_interval", 10),
        save_steps=config.get("save_interval", 100),
        eval_steps=config.get("eval_interval", 50) if valid_dataset else None,
        evaluation_strategy="steps" if valid_dataset else "no",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}...")
    if output_dir.startswith("gs://"):
        # For GCS, save locally first then upload
        local_output = Path("/tmp/sft_model")
        local_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_output)
        tokenizer.save_pretrained(local_output)

        # Validate saved model
        print("\nValidating saved model...")
        try:
            from transformers import AutoModelForCausalLM
            test_model = AutoModelForCausalLM.from_pretrained(local_output)
            test_input = tokenizer("Hello, how are you?", return_tensors="pt")
            test_output = test_model.generate(**test_input, max_length=20)
            test_text = tokenizer.decode(test_output[0], skip_special_tokens=True)
            assert len(test_text) > 0, "Model generated empty output"
            print(f"✓ Model validation passed. Test output: {test_text[:50]}...")
        except Exception as e:
            print(f"✗ Model validation FAILED: {e}")
            raise RuntimeError("Model validation failed - not uploading to GCS") from e

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

        # Validate saved model
        print("\nValidating saved model...")
        try:
            from transformers import AutoModelForCausalLM
            test_model = AutoModelForCausalLM.from_pretrained(output_path)
            test_input = tokenizer("Hello, how are you?", return_tensors="pt")
            test_output = test_model.generate(**test_input, max_length=20)
            test_text = tokenizer.decode(test_output[0], skip_special_tokens=True)
            assert len(test_text) > 0, "Model generated empty output"
            print(f"✓ Model validation passed. Test output: {test_text[:50]}...")
        except Exception as e:
            print(f"✗ Model validation FAILED: {e}")
            raise RuntimeError("Model validation failed") from e

        print(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SFT model with QLoRA")
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

    # Extract SFT config from experiment config
    if "sft" in full_config:
        sft_config = full_config["sft"]
        # Add global settings
        if "global" in full_config:
            sft_config.update(full_config["global"])
    else:
        # Assume it's a standalone SFT config
        sft_config = full_config

    # Train
    train_sft(sft_config, args.train_data, args.valid_data, args.output)

    print("\n" + "="*60)
    print("✓ SFT Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

