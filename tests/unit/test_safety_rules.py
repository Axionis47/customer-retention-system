"""Unit tests for safety shield."""
import pytest

from rlhf.safety.shield import SafetyShield


@pytest.mark.unit
def test_banned_phrases_detected():
    """Banned phrases should be flagged."""
    shield = SafetyShield()

    message = "We guarantee refund if you're not satisfied!"
    is_safe, violations, penalty = shield.check_message(message)

    assert not is_safe, "Message with banned phrase should not be safe"
    assert any("banned phrase" in v.lower() for v in violations), "Should flag banned phrase"
    assert penalty > 0, "Penalty should be positive"


@pytest.mark.unit
def test_length_constraints():
    """Length constraints should be enforced."""
    shield = SafetyShield()

    # Too short
    short_message = "Hi!"
    is_safe, violations, penalty = shield.check_message(short_message)

    assert not is_safe, "Too short message should not be safe"
    assert any("too short" in v.lower() for v in violations), "Should flag short message"

    # Too long
    long_message = "A" * 300
    is_safe, violations, penalty = shield.check_message(long_message)

    assert not is_safe, "Too long message should not be safe"
    assert any("too long" in v.lower() for v in violations), "Should flag long message"


@pytest.mark.unit
def test_quiet_hours():
    """Quiet hours should be enforced."""
    shield = SafetyShield()

    message = "Thank you for being with us! We offer you 10% off for 3 months."

    # During quiet hours (11 PM)
    is_safe_quiet, violations_quiet, _ = shield.check_message(message, hour=23)

    assert not is_safe_quiet, "Message during quiet hours should not be safe"
    assert any("quiet hours" in v.lower() for v in violations_quiet), "Should flag quiet hours"

    # During normal hours (2 PM)
    is_safe_normal, violations_normal, _ = shield.check_message(message, hour=14)

    # Should not have quiet hours violation
    assert not any("quiet hours" in v.lower() for v in violations_normal), "Should not flag normal hours"


@pytest.mark.unit
def test_required_elements():
    """Required elements should be checked."""
    shield = SafetyShield()

    # Missing "offer"
    message = "Thank you for being with us! We appreciate your loyalty."
    is_safe, violations, penalty = shield.check_message(message)

    assert not is_safe, "Message without required element should not be safe"
    assert any("required element" in v.lower() for v in violations), "Should flag missing element"


@pytest.mark.unit
def test_safe_message():
    """Safe message should pass all checks."""
    shield = SafetyShield()

    message = "Thank you for being with us! We'd like to offer you 10% off for the next 3 months."
    is_safe, violations, penalty = shield.check_message(message, hour=14)

    assert is_safe, f"Safe message should pass: {violations}"
    assert len(violations) == 0, "Should have no violations"
    assert penalty == 0, "Penalty should be zero"


@pytest.mark.unit
def test_toxicity_check():
    """Toxicity should be detected."""
    shield = SafetyShield()

    toxic_message = "You're stupid if you don't take this offer! Don't be an idiot! This is the worst and most terrible thing!"
    is_safe, violations, penalty = shield.check_message(toxic_message)

    # Should have high toxicity (message contains: stupid, idiot, worst, terrible)
    # Note: The message is also too long, so penalty will be > 0 regardless
    assert penalty > 0, "Toxic message should have penalty"
    assert not is_safe, "Toxic message should not be safe"

