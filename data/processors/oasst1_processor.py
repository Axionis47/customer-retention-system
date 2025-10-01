"""Process OASST1 dataset for SFT training."""
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from data.catalog_manager import DataCatalog


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_conversations(dataset, config: dict) -> list:
    """Extract prompt-response pairs from OASST1."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fast tokenizer for counting
    
    text_limits = config["text_limits"]
    max_prompt_tokens = text_limits["max_prompt_tokens"]
    max_response_tokens = text_limits["max_response_tokens"]
    min_length = text_limits["min_text_length"]
    
    pairs = []
    
    # OASST1 has a tree structure - we need to extract message threads
    # Group by message_tree_id and role
    message_trees = {}
    
    for example in dataset:
        tree_id = example.get("message_tree_id")
        if tree_id not in message_trees:
            message_trees[tree_id] = []
        message_trees[tree_id].append(example)
    
    # Extract prompt-response pairs from each tree
    for tree_id, messages in message_trees.items():
        # Sort by created_date to get conversation order
        messages = sorted(messages, key=lambda x: x.get("created_date", ""))
        
        # Find prompter-assistant pairs
        for i in range(len(messages) - 1):
            if messages[i].get("role") == "prompter" and messages[i+1].get("role") == "assistant":
                prompt = messages[i].get("text", "").strip()
                response = messages[i+1].get("text", "").strip()
                
                # Filter by length and token count
                if len(prompt) < min_length or len(response) < min_length:
                    continue
                
                if prompt and response:
                    prompt_tokens = count_tokens(prompt, tokenizer)
                    response_tokens = count_tokens(response, tokenizer)
                    
                    if prompt_tokens <= max_prompt_tokens and response_tokens <= max_response_tokens:
                        pairs.append({
                            "prompt": prompt,
                            "response": response
                        })
    
    return pairs


def process_oasst1(config: dict, force: bool = False):
    """Process OASST1 dataset to JSONL."""
    processed_dir = Path("data/processed/oasst1")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / "sft_train.jsonl"
    valid_path = processed_dir / "sft_valid.jsonl"
    
    catalog = DataCatalog()
    
    # Check if already processed
    if not force and catalog.exists("oasst1_train") and train_path.exists():
        print(f"OASST1 already processed: {train_path}")
        print(catalog.get("oasst1_train"))
        return
    
    # Load from HuggingFace
    print("Loading OASST1 from HuggingFace...")
    source_config = config["sources"]["oasst1"]
    dataset = load_dataset(source_config["hf_dataset"], split=source_config["split"])
    
    print(f"Loaded {len(dataset)} examples")
    
    # Extract conversations
    print("Extracting prompt-response pairs...")
    pairs = extract_conversations(dataset, config)
    
    print(f"Extracted {len(pairs)} pairs")
    
    # Cap and split
    caps = config["caps"]
    max_train = caps["oasst1_max_rows"]
    max_valid = caps["oasst1_valid_rows"]

    # Shuffle with fixed seed
    import random
    random.seed(config["splits"]["seed"])
    random.shuffle(pairs)

    # Split: first take validation, then training (to ensure we always have validation)
    total_needed = min(len(pairs), max_train + max_valid)
    valid_pairs = pairs[:max_valid]
    train_pairs = pairs[max_valid:total_needed]

    print(f"Split: train={len(train_pairs)}, valid={len(valid_pairs)}")
    
    # Validate
    validation_config = config["validation"]["oasst1"]
    assert len(train_pairs) >= validation_config["min_pairs"], f"Too few training pairs: {len(train_pairs)}"
    
    # Write JSONL
    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    with open(valid_path, "w", encoding="utf-8") as f:
        for pair in valid_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"Saved: {train_path} ({len(train_pairs)} pairs)")
    print(f"Saved: {valid_path} ({len(valid_pairs)} pairs)")
    
    # Register in catalog
    catalog.register("oasst1_train", train_path, {"pairs": len(train_pairs)})
    catalog.register("oasst1_valid", valid_path, {"pairs": len(valid_pairs)})
    
    print("✓ OASST1 processing complete")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process OASST1 dataset")
    parser.add_argument("--force", action="store_true", help="Force reprocess")
    args = parser.parse_args()
    
    # Load config
    with open("ops/configs/data_config.yaml") as f:
        config = yaml.safe_load(f)
    
    if not config["enable"]["oasst1"]:
        print("OASST1 dataset disabled in config")
        return
    
    process_oasst1(config, force=args.force)


if __name__ == "__main__":
    main()

