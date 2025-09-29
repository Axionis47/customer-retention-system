"""Unit tests for adaptive KL controller."""
import pytest

from rlhf.ppo_text import AdaptiveKLController


@pytest.mark.unit
def test_kl_increases_when_high():
    """Beta should increase when KL is above target."""
    controller = AdaptiveKLController(
        init_beta=0.1,
        target_kl=0.01,
    )

    initial_beta = controller.get_beta()

    # High KL (above target * 1.5)
    controller.update(current_kl=0.02)

    updated_beta = controller.get_beta()

    assert updated_beta > initial_beta, "Beta should increase when KL is high"


@pytest.mark.unit
def test_kl_decreases_when_low():
    """Beta should decrease when KL is below target."""
    controller = AdaptiveKLController(
        init_beta=0.5,
        target_kl=0.01,
    )

    initial_beta = controller.get_beta()

    # Low KL (below target * 0.5)
    controller.update(current_kl=0.001)

    updated_beta = controller.get_beta()

    assert updated_beta < initial_beta, "Beta should decrease when KL is low"


@pytest.mark.unit
def test_kl_clamped():
    """Beta should be clamped to valid range."""
    controller = AdaptiveKLController(
        init_beta=0.1,
        target_kl=0.01,
    )

    # Very high KL (should clamp to max)
    for _ in range(100):
        controller.update(current_kl=1.0)

    beta = controller.get_beta()
    assert beta <= 1.0, "Beta should be clamped to max 1.0"

    # Reset and test min
    controller = AdaptiveKLController(init_beta=0.1, target_kl=0.01)

    # Very low KL (should clamp to min)
    for _ in range(100):
        controller.update(current_kl=0.0)

    beta = controller.get_beta()
    assert beta >= 0.01, "Beta should be clamped to min 0.01"


@pytest.mark.unit
def test_kl_stable_at_target():
    """Beta should be relatively stable when KL is near target."""
    controller = AdaptiveKLController(
        init_beta=0.1,
        target_kl=0.01,
    )

    initial_beta = controller.get_beta()

    # KL at target
    for _ in range(10):
        controller.update(current_kl=0.01)

    final_beta = controller.get_beta()

    # Should not change much
    assert abs(final_beta - initial_beta) < 0.05, "Beta should be stable near target"

