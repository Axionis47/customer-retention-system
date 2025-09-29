"""Plotting utilities for evaluation."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def plot_pacing(
    budget_history: List[float],
    episode_length: int,
    output_path: str = "pacing_plot.png",
):
    """
    Plot budget pacing over episode.

    Args:
        budget_history: List of budget values over time
        episode_length: Total episode length
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(len(budget_history))
    ax.plot(steps, budget_history, label="Budget Remaining", linewidth=2)

    # Ideal pacing (linear)
    initial_budget = budget_history[0] if budget_history else 1000
    ideal_pacing = np.linspace(initial_budget, 0, episode_length)
    ax.plot(ideal_pacing, label="Ideal Linear Pacing", linestyle="--", alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Budget Remaining")
    ax.set_title("Budget Pacing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Pacing plot saved to {output_path}")


def plot_nrr_comparison(
    policy_names: List[str],
    nrr_values: List[float],
    output_path: str = "nrr_comparison.png",
):
    """
    Plot NRR comparison across policies.

    Args:
        policy_names: List of policy names
        nrr_values: List of NRR values (%)
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(policy_names))
    bars = ax.bar(x, nrr_values, color="steelblue", alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Net Revenue Retention (%)")
    ax.set_title("NRR Comparison Across Policies")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"NRR comparison plot saved to {output_path}")


def plot_stress_test_results(
    results: Dict[str, List[float]],
    x_label: str,
    y_label: str = "Average Reward",
    title: str = "Stress Test Results",
    output_path: str = "stress_test.png",
):
    """
    Plot stress test results.

    Args:
        results: Dict with x and y values
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_key = list(results.keys())[0]
    y_key = list(results.keys())[1]

    x_values = results[x_key]
    y_values = results[y_key]

    ax.plot(x_values, y_values, marker="o", linewidth=2, markersize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Stress test plot saved to {output_path}")


def plot_training_curve(
    rewards: List[float],
    window: int = 10,
    output_path: str = "training_curve.png",
):
    """
    Plot training reward curve with moving average.

    Args:
        rewards: List of episode rewards
        window: Moving average window size
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = np.arange(len(rewards))

    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label="Episode Reward")

    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            moving_avg,
            linewidth=2,
            label=f"Moving Avg (window={window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Training curve saved to {output_path}")

