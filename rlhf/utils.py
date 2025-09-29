"""Utility functions for RLHF pipeline."""
import json
from pathlib import Path
from typing import Dict, List

import torch


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def format_prompt(customer_context: Dict, offer_pct: float) -> str:
    """
    Format prompt for message generation.

    Args:
        customer_context: Dict with customer info
        offer_pct: Discount percentage (0-1)

    Returns:
        Formatted prompt string
    """
    prompt = f"""Generate a concise retention message for a customer.

Customer Context:
- Tenure: {customer_context.get('tenure_months', 'N/A')} months
- Monthly Spend: ${customer_context.get('monthly_spend', 'N/A')}
- Churn Risk: {customer_context.get('churn_risk', 'N/A')}

Offer: {offer_pct * 100:.0f}% discount for 3 months

Message (max 150 characters):"""

    return prompt


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

