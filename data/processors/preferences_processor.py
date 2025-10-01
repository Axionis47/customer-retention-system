"""Process SHP-2 and HH-RLHF preference datasets."""
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from data.catalog_manager import DataCatalog


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_shp2_pairs(dataset, config: dict, max_pairs: int) -> list:
    """Extract preference pairs from SHP-2."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    text_limits = config["text_limits"]
    max_prompt_tokens = text_limits["max_prompt_tokens"]
    max_response_tokens = text_limits["max_response_tokens"]
    min_length = text_limits["min_text_length"]
    
    pairs = []
    
    for example in dataset:
        if len(pairs) >= max_pairs:
            break
        
        # SHP-2 format: post_id, history, human_ref_A, human_ref_B, labels, score_A, score_B
        history = example.get("history", "")
        chosen = example.get("human_ref_A", "")
        rejected = example.get("human_ref_B", "")
        score_a = example.get("score_A", 0)
        score_b = example.get("score_B", 0)
        
        # Determine which is chosen based on scores
        if score_a < score_b:
            chosen, rejected = rejected, chosen
        
        # Filter
        if len(history) < min_length or len(chosen) < min_length or len(rejected) < min_length:
            continue
        
        if history and chosen and rejected:
            prompt_tokens = count_tokens(history, tokenizer)
            chosen_tokens = count_tokens(chosen, tokenizer)
            rejected_tokens = count_tokens(rejected, tokenizer)
            
            if (prompt_tokens <= max_prompt_tokens and 
                chosen_tokens <= max_response_tokens and 
                rejected_tokens <= max_response_tokens):
                pairs.append({
                    "prompt": history.strip(),
                    "chosen": chosen.strip(),
                    "rejected": rejected.strip(),
                    "source": "shp2"
                })
    
    return pairs


def extract_hh_pairs(dataset, config: dict, max_pairs: int) -> list:
    """Extract preference pairs from HH-RLHF."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    text_limits = config["text_limits"]
    max_prompt_tokens = text_limits["max_prompt_tokens"]
    max_response_tokens = text_limits["max_response_tokens"]
    min_length = text_limits["min_text_length"]
    
    pairs = []
    
    for example in dataset:
        if len(pairs) >= max_pairs:
            break
        
        # HH-RLHF format: chosen, rejected (full conversations)
        chosen_conv = example.get("chosen", "")
        rejected_conv = example.get("rejected", "")
        
        # Extract prompt (everything before last Assistant response)
        # Format: "\n\nHuman: ... \n\nAssistant: ..."
        if "\n\nAssistant:" in chosen_conv:
            parts = chosen_conv.split("\n\nAssistant:")
            prompt = parts[0].strip()
            chosen_response = parts[-1].strip()
        else:
            continue
        
        if "\n\nAssistant:" in rejected_conv:
            rejected_response = rejected_conv.split("\n\nAssistant:")[-1].strip()
        else:
            continue
        
        # Filter
        if (len(prompt) < min_length or 
            len(chosen_response) < min_length or 
            len(rejected_response) < min_length):
            continue
        
        if prompt and chosen_response and rejected_response:
            prompt_tokens = count_tokens(prompt, tokenizer)
            chosen_tokens = count_tokens(chosen_response, tokenizer)
            rejected_tokens = count_tokens(rejected_response, tokenizer)
            
            if (prompt_tokens <= max_prompt_tokens and 
                chosen_tokens <= max_response_tokens and 
                rejected_tokens <= max_response_tokens):
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "source": "hh"
                })
    
    return pairs


def process_preferences(config: dict, force: bool = False):
    """Process SHP-2 and HH-RLHF datasets to combined JSONL."""
    processed_dir = Path("data/processed/preferences")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_path = processed_dir / "pairs.jsonl"
    valid_path = processed_dir / "pairs_valid.jsonl"
    probe_path = processed_dir / "hh_probe.jsonl"
    
    catalog = DataCatalog()
    
    # Check if already processed
    if not force and catalog.exists("preferences_train") and pairs_path.exists():
        print(f"Preferences already processed: {pairs_path}")
        print(catalog.get("preferences_train"))
        return
    
    caps = config["caps"]
    all_pairs = []
    
    # Load SHP-2
    if config["enable"]["shp2"]:
        print("Loading SHP-2 from HuggingFace...")
        shp2_config = config["sources"]["shp2"]
        shp2_dataset = load_dataset(shp2_config["hf_dataset"], split=shp2_config["split"])
        print(f"Loaded {len(shp2_dataset)} SHP-2 examples")
        
        print("Extracting SHP-2 pairs...")
        shp2_pairs = extract_shp2_pairs(shp2_dataset, config, caps["shp2_max"])
        print(f"Extracted {len(shp2_pairs)} SHP-2 pairs")
        all_pairs.extend(shp2_pairs)
    
    # Load HH-RLHF
    hh_pairs_for_probe = []
    if config["enable"]["hh"]:
        print("Loading HH-RLHF from HuggingFace...")
        hh_config = config["sources"]["hh_rlhf"]
        hh_dataset = load_dataset(hh_config["hf_dataset"], split=hh_config["split"])
        print(f"Loaded {len(hh_dataset)} HH-RLHF examples")
        
        print("Extracting HH-RLHF pairs...")
        hh_pairs = extract_hh_pairs(hh_dataset, config, caps["hh_max"])
        print(f"Extracted {len(hh_pairs)} HH-RLHF pairs")
        
        # Save some for probe
        hh_pairs_for_probe = hh_pairs[:caps["prefs_hh_probe"]]
        all_pairs.extend(hh_pairs)
    
    # Shuffle and cap
    import random
    random.seed(config["splits"]["seed"])
    random.shuffle(all_pairs)
    
    max_total = caps["prefs_max_pairs"]
    all_pairs = all_pairs[:max_total]
    
    print(f"Total pairs after cap: {len(all_pairs)}")
    
    # Split train/valid
    max_valid = caps["prefs_valid_pairs"]
    train_pairs = all_pairs[:-max_valid]
    valid_pairs = all_pairs[-max_valid:]
    
    print(f"Split: train={len(train_pairs)}, valid={len(valid_pairs)}")
    
    # Validate
    validation_config = config["validation"]["preferences"]
    assert len(train_pairs) >= validation_config["min_pairs"], f"Too few training pairs: {len(train_pairs)}"
    
    # Write JSONL
    with open(pairs_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    with open(valid_path, "w", encoding="utf-8") as f:
        for pair in valid_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Write HH probe
    if hh_pairs_for_probe:
        with open(probe_path, "w", encoding="utf-8") as f:
            for pair in hh_pairs_for_probe:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved HH probe: {probe_path} ({len(hh_pairs_for_probe)} pairs)")
    
    print(f"Saved: {pairs_path} ({len(train_pairs)} pairs)")
    print(f"Saved: {valid_path} ({len(valid_pairs)} pairs)")
    
    # Compute source distribution
    source_dist = {}
    for pair in train_pairs:
        source = pair["source"]
        source_dist[source] = source_dist.get(source, 0) + 1
    
    print(f"Source distribution: {source_dist}")
    
    # Register in catalog
    catalog.register("preferences_train", pairs_path, {"pairs": len(train_pairs), "source_dist": source_dist})
    catalog.register("preferences_valid", valid_path, {"pairs": len(valid_pairs)})
    if hh_pairs_for_probe:
        catalog.register("preferences_hh_probe", probe_path, {"pairs": len(hh_pairs_for_probe)})
    
    print("✓ Preferences processing complete")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process preference datasets")
    parser.add_argument("--force", action="store_true", help="Force reprocess")
    args = parser.parse_args()
    
    # Load config
    with open("ops/configs/data_config.yaml") as f:
        config = yaml.safe_load(f)
    
    if not (config["enable"]["shp2"] or config["enable"]["hh"]):
        print("All preference datasets disabled in config")
        return
    
    process_preferences(config, force=args.force)


if __name__ == "__main__":
    main()

