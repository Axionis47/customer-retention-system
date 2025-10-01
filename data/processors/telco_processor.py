"""Process IBM Telco Customer Churn dataset."""
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from data.catalog_manager import DataCatalog


def download_telco(config: dict, force: bool = False) -> Path:
    """Download Telco dataset from Kaggle."""
    raw_dir = Path("data/raw/telco")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    source_config = config["sources"]["telco"]
    csv_path = raw_dir / source_config["filename"]
    
    if csv_path.exists() and not force:
        print(f"Telco CSV already exists: {csv_path}")
        return csv_path
    
    # Check for Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Please:\n"
            "1. Go to https://www.kaggle.com/settings/account\n"
            "2. Click 'Create New API Token'\n"
            "3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
    
    # Download using kaggle CLI
    import subprocess
    
    dataset_slug = source_config["kaggle_dataset"]
    print(f"Downloading {dataset_slug} from Kaggle...")
    
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir), "--unzip"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found after download: {csv_path}")
    
    print(f"Downloaded to: {csv_path}")
    return csv_path


def process_telco(config: dict, force: bool = False):
    """Process Telco dataset to parquet with splits."""
    processed_dir = Path("data/processed/telco")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / "telco.parquet"
    catalog = DataCatalog()
    
    # Check if already processed
    if not force and catalog.exists("telco") and output_path.exists():
        print(f"Telco already processed: {output_path}")
        print(catalog.get("telco"))
        return
    
    # Download if needed
    raw_csv = download_telco(config, force=force)
    
    # Load and process
    print("Processing Telco dataset...")
    df = pd.read_csv(raw_csv)
    
    # Coerce TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with NA in TotalCharges
    initial_rows = len(df)
    df = df.dropna(subset=["TotalCharges"])
    print(f"Dropped {initial_rows - len(df)} rows with NA in TotalCharges")
    
    # Map Churn to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    
    # Keep only required columns
    keep_cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "Churn"]
    df = df[keep_cols]
    
    # Validate
    validation_config = config["validation"]["telco"]
    assert len(df) >= validation_config["min_rows"], f"Too few rows: {len(df)}"
    
    churn_rate = df["Churn"].mean()
    assert validation_config["churn_rate_min"] <= churn_rate <= validation_config["churn_rate_max"], \
        f"Churn rate {churn_rate:.3f} out of expected range"
    
    # Save full dataset
    df.to_parquet(output_path, index=False)
    print(f"Saved full dataset: {output_path} ({len(df)} rows)")
    
    # Create splits (80/10/10)
    seed = config["splits"]["seed"]
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["Churn"])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["Churn"])
    
    train_path = processed_dir / "telco_train.parquet"
    valid_path = processed_dir / "telco_valid.parquet"
    test_path = processed_dir / "telco_test.parquet"
    
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
        "churn_rate": float(churn_rate),
        "churn_count": int(df["Churn"].sum()),
    }
    
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Register in catalog
    catalog.register("telco", output_path, {"stats": stats})
    catalog.register("telco_train", train_path)
    catalog.register("telco_valid", valid_path)
    catalog.register("telco_test", test_path)
    
    print("✓ Telco processing complete")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Telco dataset")
    parser.add_argument("--force", action="store_true", help="Force redownload and reprocess")
    args = parser.parse_args()
    
    # Load config
    with open("ops/configs/data_config.yaml") as f:
        config = yaml.safe_load(f)
    
    if not config["enable"]["telco"]:
        print("Telco dataset disabled in config")
        return
    
    process_telco(config, force=args.force)


if __name__ == "__main__":
    main()

