"""Train churn risk prediction model."""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def load_data(data_path: str) -> pd.DataFrame:
    """Load churn training data from CSV or GCS."""
    if data_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path, "r") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(data_path)
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    """Feature engineering for churn prediction."""
    # Encode contract type
    contract_map = {"month-to-month": 0, "one-year": 1, "two-year": 2}
    df["contract_encoded"] = df["contract_type"].map(contract_map).fillna(0)

    # Features
    feature_cols = ["tenure_months", "monthly_spend", "support_tickets", "contract_encoded"]
    X = df[feature_cols].values
    y = df["churned"].values

    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val, seed: int = 42):
    """Train XGBoost churn model."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="auc",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    print(f"Churn Model AUC: {auc:.4f}")
    print(report)

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn_train.csv", help="Path to training data")
    parser.add_argument("--output", default="models/risk_accept/artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} samples")

    # Preprocess
    X, y, feature_cols = preprocess(df)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    print("Training churn model...")
    model = train_model(X_train, y_train, X_val, y_val, seed=args.seed)

    # Evaluate
    print("Evaluating on test set...")
    auc = evaluate(model, X_test, y_test)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "churn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "auc": auc}, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

