"""Business metrics for retention policy evaluation."""
import numpy as np
from typing import Dict, List


def compute_nrr(
    retained_revenue: float,
    initial_revenue: float,
) -> float:
    """
    Compute Net Revenue Retention (NRR).

    NRR = (retained_revenue / initial_revenue) * 100
    """
    if initial_revenue == 0:
        return 0.0
    return (retained_revenue / initial_revenue) * 100


def compute_retention_uplift(
    retention_rate_treatment: float,
    retention_rate_control: float,
) -> float:
    """
    Compute retention uplift.

    Uplift = (treatment_rate - control_rate) / control_rate * 100
    """
    if retention_rate_control == 0:
        return 0.0
    return ((retention_rate_treatment - retention_rate_control) / retention_rate_control) * 100


def compute_roi(
    incremental_revenue: float,
    total_cost: float,
) -> float:
    """
    Compute Return on Investment (ROI).

    ROI = (incremental_revenue - total_cost) / total_cost * 100
    """
    if total_cost == 0:
        return 0.0
    return ((incremental_revenue - total_cost) / total_cost) * 100


def compute_violation_rate(
    num_violations: int,
    total_actions: int,
) -> float:
    """Compute constraint violation rate."""
    if total_actions == 0:
        return 0.0
    return (num_violations / total_actions) * 100


def compute_contact_rate(
    num_contacts: int,
    total_customers: int,
) -> float:
    """Compute contact rate."""
    if total_customers == 0:
        return 0.0
    return (num_contacts / total_customers) * 100


class BusinessMetricsTracker:
    """Track business metrics over episodes."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.num_contacts = 0
        self.num_retentions = 0
        self.num_violations = 0
        self.num_customers = 0

    def update(
        self,
        revenue: float,
        cost: float,
        contacted: bool,
        retained: bool,
        violated: bool,
    ):
        """Update metrics with episode results."""
        self.total_revenue += revenue
        self.total_cost += cost
        self.num_contacts += int(contacted)
        self.num_retentions += int(retained)
        self.num_violations += int(violated)
        self.num_customers += 1

    def get_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics."""
        metrics = {
            "total_revenue": self.total_revenue,
            "total_cost": self.total_cost,
            "net_revenue": self.total_revenue - self.total_cost,
            "contact_rate": compute_contact_rate(self.num_contacts, self.num_customers),
            "retention_rate": (
                self.num_retentions / self.num_customers * 100 if self.num_customers > 0 else 0.0
            ),
            "violation_rate": compute_violation_rate(self.num_violations, self.num_customers),
            "avg_cost_per_contact": (
                self.total_cost / self.num_contacts if self.num_contacts > 0 else 0.0
            ),
            "roi": compute_roi(self.total_revenue, self.total_cost),
        }
        return metrics

