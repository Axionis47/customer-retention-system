"""Calibrate probability predictions using isotonic regression."""
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--churn-model",
        default="models/risk_accept/artifacts/churn_model.pkl",
        help="Path to churn model",
    )
    parser.add_argument(
        "--accept-model",
        default="models/risk_accept/artifacts/accept_model.pkl",
        help="Path to acceptance model",
    )
    parser.add_argument("--output", default="models/risk_accept/artifacts", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    with open(args.churn_model, "rb") as f:
        churn_data = pickle.load(f)

    with open(args.accept_model, "rb") as f:
        accept_data = pickle.load(f)

    # Create simple isotonic calibrators (placeholder - would need validation data in production)
    # For now, just save identity calibrators
    churn_calibrator = IsotonicRegression(out_of_bounds="clip")
    accept_calibrator = IsotonicRegression(out_of_bounds="clip")

    # Fit on dummy data (in production, use validation set)
    dummy_x = np.linspace(0, 1, 100)
    churn_calibrator.fit(dummy_x, dummy_x)
    accept_calibrator.fit(dummy_x, dummy_x)

    # Save calibrators
    calibrator_path = output_dir / "calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(
            {
                "churn_calibrator": churn_calibrator,
                "accept_calibrator": accept_calibrator,
            },
            f,
        )

    print(f"Calibrators saved to {calibrator_path}")
    print("Note: Using identity calibration. In production, fit on validation data.")


if __name__ == "__main__":
    main()

