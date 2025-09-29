"""Generate demo training data locally."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn training data."""
    np.random.seed(seed)

    data = {
        "customer_id": [f"C{i:05d}" for i in range(n_samples)],
        "tenure_months": np.random.randint(1, 60, n_samples),
        "monthly_spend": np.random.uniform(20, 200, n_samples),
        "support_tickets": np.random.poisson(2, n_samples),
        "contract_type": np.random.choice(
            ["month-to-month", "one-year", "two-year"],
            n_samples,
            p=[0.5, 0.3, 0.2],
        ),
    }

    df = pd.DataFrame(data)

    # Generate churn labels (higher risk for short tenure, high tickets)
    churn_prob = 1 / (1 + np.exp(-(
        -2.0
        + 0.05 * (30 - df["tenure_months"])
        + 0.3 * df["support_tickets"]
        - 0.01 * df["monthly_spend"]
    )))

    df["churned"] = (np.random.random(n_samples) < churn_prob).astype(int)

    return df


def generate_acceptance_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic acceptance training data."""
    np.random.seed(seed)

    data = {
        "customer_id": [f"C{i:05d}" for i in range(n_samples)],
        "offer_pct": np.random.choice([0.0, 0.05, 0.10, 0.20], n_samples),
        "churn_risk": np.random.beta(2, 5, n_samples),
        "tenure_months": np.random.randint(1, 60, n_samples),
        "monthly_spend": np.random.uniform(20, 200, n_samples),
    }

    df = pd.DataFrame(data)

    # Generate acceptance labels (higher for high churn risk, high offer)
    accept_prob = 1 / (1 + np.exp(-(
        -1.0
        + 3.0 * df["offer_pct"]
        + 2.0 * df["churn_risk"]
        + 0.01 * df["tenure_months"]
    )))

    df["accepted"] = (np.random.random(n_samples) < accept_prob).astype(int)

    return df


def generate_rlhf_pairs(n_pairs: int = 100, seed: int = 42) -> list:
    """Generate synthetic RLHF preference pairs."""
    np.random.seed(seed)

    pairs = []

    templates_good = [
        "As a valued customer, we'd like to offer you {offer}% off for 3 months. Thank you for your loyalty!",
        "We appreciate your {tenure} months with us. Enjoy {offer}% off your next 3 months!",
        "Thank you for being with us! We're offering you {offer}% off for the next quarter.",
    ]

    templates_bad = [
        "URGENT! You must accept this {offer}% discount NOW or lose it forever!",
        "Last chance! {offer}% off but only if you act immediately!",
        "Don't miss out! {offer}% discount expires soon! Act now!",
    ]

    for i in range(n_pairs):
        tenure = np.random.randint(1, 60)
        offer = np.random.choice([5, 10, 20])

        prompt = f"Generate a retention message for a customer with {tenure} months tenure. Offer: {offer}% discount."

        chosen = np.random.choice(templates_good).format(offer=offer, tenure=tenure)
        rejected = np.random.choice(templates_bad).format(offer=offer, tenure=tenure)

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "response": chosen,  # For SFT
        })

    return pairs


def main():
    """Generate all demo datasets."""
    print("Generating demo datasets...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Churn data
    print("Generating churn training data...")
    churn_df = generate_churn_data(n_samples=1000)
    churn_df.to_csv(data_dir / "churn_train.csv", index=False)
    print(f"Saved {len(churn_df)} samples to data/churn_train.csv")

    # Acceptance data
    print("Generating acceptance training data...")
    accept_df = generate_acceptance_data(n_samples=1000)
    accept_df.to_csv(data_dir / "accept_train.csv", index=False)
    print(f"Saved {len(accept_df)} samples to data/accept_train.csv")

    # RLHF pairs
    print("Generating RLHF preference pairs...")
    pairs = generate_rlhf_pairs(n_pairs=100)
    with open(data_dir / "rlhf_pairs.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved {len(pairs)} pairs to data/rlhf_pairs.jsonl")

    print("Done!")


if __name__ == "__main__":
    main()

