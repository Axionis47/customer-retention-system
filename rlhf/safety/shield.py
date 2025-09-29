"""Safety shield for message generation."""
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class SafetyShield:
    """
    Safety shield to filter and penalize unsafe messages.

    Checks:
        - Banned phrases
        - Length constraints
        - Quiet hours (if timestamp provided)
        - Toxicity (placeholder)
        - Forbidden patterns (PII)
    """

    def __init__(self, rules_path: str = "rlhf/safety/rules.yaml"):
        """Load safety rules."""
        rules_file = Path(rules_path)
        if rules_file.exists():
            with open(rules_file) as f:
                self.rules = yaml.safe_load(f)
        else:
            # Default rules
            self.rules = {
                "banned_phrases": ["guarantee refund", "illegal", "guaranteed"],
                "max_length": 200,
                "min_length": 20,
                "quiet_hours": {"start": 22, "end": 8},
                "toxicity_threshold": 0.7,
                "required_elements": ["offer"],
                "forbidden_patterns": [],
            }

    def check_message(self, message: str, hour: int = 12) -> Tuple[bool, List[str], float]:
        """
        Check message against safety rules.

        Args:
            message: Generated message text
            hour: Hour of day (0-23) for quiet hours check

        Returns:
            is_safe: Whether message passes all checks
            violations: List of violation descriptions
            penalty: Numeric penalty score
        """
        violations = []
        penalty = 0.0

        # Length check
        if len(message) > self.rules["max_length"]:
            violations.append(f"Message too long ({len(message)} > {self.rules['max_length']})")
            penalty += 1.0

        if len(message) < self.rules["min_length"]:
            violations.append(f"Message too short ({len(message)} < {self.rules['min_length']})")
            penalty += 1.0

        # Banned phrases
        message_lower = message.lower()
        for phrase in self.rules["banned_phrases"]:
            if phrase.lower() in message_lower:
                violations.append(f"Banned phrase: '{phrase}'")
                penalty += 2.0

        # Quiet hours
        quiet_start = self.rules["quiet_hours"]["start"]
        quiet_end = self.rules["quiet_hours"]["end"]
        if quiet_start > quiet_end:  # Wraps midnight
            in_quiet_hours = hour >= quiet_start or hour < quiet_end
        else:
            in_quiet_hours = quiet_start <= hour < quiet_end

        if in_quiet_hours:
            violations.append(f"Quiet hours violation (hour={hour})")
            penalty += 0.5

        # Required elements
        for element in self.rules.get("required_elements", []):
            if element.lower() not in message_lower:
                violations.append(f"Missing required element: '{element}'")
                penalty += 0.5

        # Forbidden patterns (PII)
        for pattern in self.rules.get("forbidden_patterns", []):
            if re.search(pattern, message):
                violations.append(f"Forbidden pattern detected: {pattern}")
                penalty += 5.0

        # Toxicity (placeholder - would use real classifier)
        toxicity_score = self._check_toxicity(message)
        if toxicity_score > self.rules["toxicity_threshold"]:
            violations.append(f"Toxicity score too high: {toxicity_score:.2f}")
            penalty += 3.0

        is_safe = len(violations) == 0

        return is_safe, violations, penalty

    def _check_toxicity(self, message: str) -> float:
        """
        Placeholder toxicity check.

        In production, use a real toxicity classifier (e.g., Perspective API, Detoxify).

        Args:
            message: Message text

        Returns:
            toxicity_score: Score in [0, 1]
        """
        # Simple heuristic: check for aggressive words
        aggressive_words = ["hate", "stupid", "idiot", "worst", "terrible", "awful"]
        message_lower = message.lower()

        count = sum(1 for word in aggressive_words if word in message_lower)
        toxicity_score = min(count * 0.2, 1.0)

        return toxicity_score

    def get_penalty(self, message: str, hour: int = 12) -> float:
        """Get penalty score for message."""
        _, _, penalty = self.check_message(message, hour)
        return penalty

