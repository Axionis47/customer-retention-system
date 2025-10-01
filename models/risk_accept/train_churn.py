"""Train churn risk prediction model on Telco dataset."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss, log_loss


def load_data(train_path: str, valid_path: str, test_path: str) -> tuple:
    """Load Telco churn data from parquet files (local or GCS)."""
    # Support GCS paths
    if train_path.startswith("/gcs/"):
        train_path = "gs://" + train_path[5:]
    if valid_path.startswith("/gcs/"):
        valid_path = "gs://" + valid_path[5:]
    if test_path.startswith("/gcs/"):
        test_path = "gs://" + test_path[5:]

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)
    return train_df, valid_df, test_df


def preprocess(df: pd.DataFrame) -> tuple:
    """Feature engineering for Telco churn prediction."""
    # Map Contract to numeric
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["Contract_encoded"] = df["Contract"].map(contract_map).fillna(0)

    # Map PaymentMethod to numeric
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
    df["PaymentMethod_encoded"] = df["PaymentMethod"].map(payment_map).fillna(0)

    # Features from Telco dataset
    feature_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract_encoded",
        "PaymentMethod_encoded"
    ]
    X = df[feature_cols].values
    y = df["Churn"].values

    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val, config: dict, seed: int = 42):
    """Train XGBoost churn model with config params."""
    params = config.get("params", {})
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=seed,
        eval_metric="auc",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    return model


def calibrate_model(model, X_cal, y_cal, method="isotonic"):
    """Calibrate model probabilities."""
    print(f"Calibrating model with {method} method...")
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def evaluate(model, X_test, y_test, exit_criteria: dict):
    """Evaluate model performance against exit criteria."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    ece = compute_ece(y_test, y_pred_proba)

    report = classification_report(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"Churn Model Evaluation")
    print(f"{'='*60}")
    print(f"AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"\n{report}")

    # Check exit criteria
    min_auc = exit_criteria.get("min_auc", 0.78)
    max_ece = exit_criteria.get("max_ece", 0.05)

    passed = auc >= min_auc and ece <= max_ece

    if passed:
        print(f"✓ EXIT CRITERIA MET: AUC={auc:.4f} >= {min_auc}, ECE={ece:.4f} <= {max_ece}")
    else:
        print(f"✗ EXIT CRITERIA FAILED:")
        if auc < min_auc:
            print(f"  - AUC={auc:.4f} < {min_auc}")
        if ece > max_ece:
            print(f"  - ECE={ece:.4f} > {max_ece}")

    metrics = {
        "auc": float(auc),
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "ece": float(ece),
        "exit_criteria_passed": bool(passed)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/experiment_exp_001_mvp.yaml", help="Experiment config")
    parser.add_argument("--train-data", default="data/processed/telco/telco_train.parquet", help="Training data")
    parser.add_argument("--valid-data", default="data/processed/telco/telco_valid.parquet", help="Validation data")
    parser.add_argument("--test-data", default="data/processed/telco/telco_test.parquet", help="Test data")
    parser.add_argument("--output", default=None, help="Output path (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_config = config["risk_model"]
    seed = config["global"]["seed"]

    # Set seed
    np.random.seed(seed)

    # Load data
    print(f"Loading Telco data...")
    print(f"  Train: {args.train_data}")
    print(f"  Valid: {args.valid_data}")
    print(f"  Test: {args.test_data}")

    train_df, valid_df, test_df = load_data(args.train_data, args.valid_data, args.test_data)
    print(f"Loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Preprocess
    X_train, y_train, feature_cols = preprocess(train_df)
    X_valid, y_valid, _ = preprocess(valid_df)
    X_test, y_test, _ = preprocess(test_df)

    print(f"Features: {feature_cols}")
    print(f"Churn rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}, test={y_test.mean():.3f}")

    # Train
    print("\nTraining churn risk model...")
    model = train_model(X_train, y_train, X_valid, y_valid, exp_config, seed=seed)

    # Calibrate
    calibrated_model = calibrate_model(model, X_valid, y_valid, method=exp_config["calibration"]["method"])

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate(calibrated_model, X_test, y_test, exp_config["exit_criteria"])

    # Save
    output_path = args.output or exp_config["output_path"]

    # Handle GCS paths
    if output_path.startswith("/gcs/"):
        output_path = "gs://" + output_path[5:]

    artifact = {
        "model": calibrated_model,
        "features": feature_cols,
        "metrics": metrics,
        "config": exp_config,
        "experiment": config["experiment"]["name"]
    }

    # Save model
    if output_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(output_path, "wb") as f:
            pickle.dump(artifact, f)
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)

    print(f"\n✓ Model saved to {output_path}")

    # Save metrics as JSON
    if output_path.startswith("gs://"):
        metrics_path = output_path.replace(".pkl", "_metrics.json")
        with fs.open(metrics_path, "w") as f:
            f.write(json.dumps(metrics, indent=2))
    else:
        output_dir = Path(output_path).parent
        metrics_path = output_dir / f"{config['experiment']['name']}_risk_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

