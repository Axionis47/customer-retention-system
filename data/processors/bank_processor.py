"""Process UCI Bank Marketing dataset."""
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from data.catalog_manager import DataCatalog


def download_bank(config: dict, force: bool = False) -> Path:
    """Download Bank Marketing dataset from UCI."""
    raw_dir = Path("data/raw/bank_marketing")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    source_config = config["sources"]["bank"]
    zip_path = raw_dir / "bank-additional.zip"
    csv_path = raw_dir / source_config["filename"]
    
    if csv_path.exists() and not force:
        print(f"Bank CSV already exists: {csv_path}")
        return csv_path
    
    # Download zip
    if not zip_path.exists() or force:
        print(f"Downloading from {source_config['url']}...")
        # Create SSL context that doesn't verify certificates (for UCI repository)
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(source_config["url"], context=ssl_context) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Downloaded to: {zip_path}")
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found after extraction: {csv_path}")
    
    print(f"Extracted to: {csv_path}")
    return csv_path


def compute_offer_level(row) -> int:
    """
    Compute offer_level proxy from campaign features.
    
    Heuristic: Higher campaign intensity + recent contact = higher offer level.
    - campaign: number of contacts during this campaign
    - pdays: days since last contact (-1 = never contacted)
    - previous: number of contacts before this campaign
    
    Returns: 0 (no offer), 1 (low), 2 (medium), 3 (high)
    """
    campaign = row.get("campaign", 1)
    pdays = row.get("pdays", 999)
    previous = row.get("previous", 0)
    
    # Score based on contact intensity
    score = 0
    
    # More campaigns = higher offer
    if campaign >= 5:
        score += 2
    elif campaign >= 3:
        score += 1
    
    # Recent contact = higher offer
    if pdays != 999 and pdays < 7:
        score += 2
    elif pdays != 999 and pdays < 30:
        score += 1
    
    # Previous contacts = higher offer
    if previous >= 2:
        score += 1
    
    # Map score to offer level (0-3)
    if score >= 4:
        return 3
    elif score >= 2:
        return 2
    elif score >= 1:
        return 1
    else:
        return 0


def process_bank(config: dict, force: bool = False):
    """Process Bank Marketing dataset to parquet with splits."""
    processed_dir = Path("data/processed/bank_marketing")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / "bank.parquet"
    catalog = DataCatalog()
    
    # Check if already processed
    if not force and catalog.exists("bank") and output_path.exists():
        print(f"Bank already processed: {output_path}")
        print(catalog.get("bank"))
        return
    
    # Download if needed
    raw_csv = download_bank(config, force=force)
    
    # Load and process
    print("Processing Bank Marketing dataset...")
    df = pd.read_csv(raw_csv, sep=";")
    
    # Map label y to binary
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype("int8")
    
    # Add offer_level proxy
    print("Computing offer_level proxy...")
    df["offer_level"] = df.apply(compute_offer_level, axis=1).astype("int8")
    
    print(f"Offer level distribution:\n{df['offer_level'].value_counts().sort_index()}")
    
    # Validate
    validation_config = config["validation"]["bank"]
    assert len(df) >= validation_config["min_rows"], f"Too few rows: {len(df)}"
    
    for col in validation_config["required_columns"]:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Save full dataset
    df.to_parquet(output_path, index=False)
    print(f"Saved full dataset: {output_path} ({len(df)} rows)")
    
    # Create splits (80/10/10)
    seed = config["splits"]["seed"]
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["y"])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["y"])
    
    train_path = processed_dir / "bank_train.parquet"
    valid_path = processed_dir / "bank_valid.parquet"
    test_path = processed_dir / "bank_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Splits: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    
    # Compute stats
    stats = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "acceptance_rate": float(df["y"].mean()),
        "accepted_count": int(df["y"].sum()),
        "offer_level_dist": df["offer_level"].value_counts().to_dict(),
    }
    
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Register in catalog
    catalog.register("bank", output_path, {"stats": stats})
    catalog.register("bank_train", train_path)
    catalog.register("bank_valid", valid_path)
    catalog.register("bank_test", test_path)
    
    print("✓ Bank Marketing processing complete")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Bank Marketing dataset")
    parser.add_argument("--force", action="store_true", help="Force redownload and reprocess")
    args = parser.parse_args()
    
    # Load config
    with open("ops/configs/data_config.yaml") as f:
        config = yaml.safe_load(f)
    
    if not config["enable"]["bank"]:
        print("Bank dataset disabled in config")
        return
    
    process_bank(config, force=args.force)


if __name__ == "__main__":
    main()

