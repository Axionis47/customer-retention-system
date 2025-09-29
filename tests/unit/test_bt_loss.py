"""Unit tests for Bradley-Terry loss."""
import torch
import pytest

from rlhf.rm_train import bradley_terry_loss


@pytest.mark.unit
def test_bt_loss_finite():
    """Bradley-Terry loss should be finite."""
    chosen_rewards = torch.tensor([1.0, 2.0, 3.0])
    rejected_rewards = torch.tensor([0.5, 1.5, 2.5])

    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)

    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"


@pytest.mark.unit
def test_bt_loss_higher_margin():
    """Higher margin between chosen and rejected should give lower loss."""
    chosen_rewards = torch.tensor([2.0, 2.0])
    rejected_rewards_close = torch.tensor([1.9, 1.9])
    rejected_rewards_far = torch.tensor([0.5, 0.5])

    loss_close = bradley_terry_loss(chosen_rewards, rejected_rewards_close)
    loss_far = bradley_terry_loss(chosen_rewards, rejected_rewards_far)

    assert loss_far < loss_close, "Larger margin should give lower loss"


@pytest.mark.unit
def test_bt_loss_with_margin():
    """Margin parameter should affect loss."""
    chosen_rewards = torch.tensor([2.0, 2.0])
    rejected_rewards = torch.tensor([1.5, 1.5])

    loss_no_margin = bradley_terry_loss(chosen_rewards, rejected_rewards, margin=0.0)
    loss_with_margin = bradley_terry_loss(chosen_rewards, rejected_rewards, margin=0.5)

    assert loss_with_margin != loss_no_margin, "Margin should affect loss"


@pytest.mark.unit
def test_bt_loss_gradient():
    """Loss should have valid gradients."""
    chosen_rewards = torch.tensor([2.0, 3.0], requires_grad=True)
    rejected_rewards = torch.tensor([1.0, 2.0], requires_grad=True)

    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
    loss.backward()

    assert chosen_rewards.grad is not None, "Chosen rewards should have gradients"
    assert rejected_rewards.grad is not None, "Rejected rewards should have gradients"
    assert torch.all(torch.isfinite(chosen_rewards.grad)), "Gradients should be finite"

