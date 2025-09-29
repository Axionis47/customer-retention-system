"""A/B arena for message quality comparison."""
import random
from typing import Callable, Dict, List, Tuple


class MessageArena:
    """
    A/B testing arena for comparing message generation policies.

    Simulates human preferences between two messages.
    """

    def __init__(self, seed: int = 42):
        """Initialize arena."""
        self.seed = seed
        random.seed(seed)
        self.comparisons: List[Dict] = []

    def compare_messages(
        self,
        message_a: str,
        message_b: str,
        context: Dict,
        judge: Callable = None,
    ) -> str:
        """
        Compare two messages and return winner.

        Args:
            message_a: First message
            message_b: Second message
            context: Customer context
            judge: Optional judge function (message, context) -> score

        Returns:
            Winner: "A", "B", or "tie"
        """
        if judge is None:
            # Simple heuristic judge
            score_a = self._heuristic_score(message_a)
            score_b = self._heuristic_score(message_b)
        else:
            score_a = judge(message_a, context)
            score_b = judge(message_b, context)

        if abs(score_a - score_b) < 0.1:
            winner = "tie"
        elif score_a > score_b:
            winner = "A"
        else:
            winner = "B"

        # Record comparison
        self.comparisons.append(
            {
                "message_a": message_a,
                "message_b": message_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "context": context,
            }
        )

        return winner

    def _heuristic_score(self, message: str) -> float:
        """
        Simple heuristic scoring for messages.

        Criteria:
            - Length (prefer 50-150 chars)
            - Mentions offer
            - Polite tone
            - No aggressive words
        """
        score = 0.0

        # Length
        length = len(message)
        if 50 <= length <= 150:
            score += 1.0
        elif length < 50:
            score += 0.5
        else:
            score += 0.3

        # Mentions offer/discount
        if any(word in message.lower() for word in ["offer", "discount", "%", "save"]):
            score += 1.0

        # Polite words
        polite_words = ["thank", "appreciate", "value", "please"]
        score += sum(0.2 for word in polite_words if word in message.lower())

        # Aggressive words (penalty)
        aggressive_words = ["must", "immediately", "urgent", "last chance"]
        score -= sum(0.3 for word in aggressive_words if word in message.lower())

        return max(0, score)

    def run_tournament(
        self,
        policy_a: Callable,
        policy_b: Callable,
        contexts: List[Dict],
        judge: Callable = None,
    ) -> Dict[str, float]:
        """
        Run tournament between two policies.

        Args:
            policy_a: First message generation policy
            policy_b: Second message generation policy
            contexts: List of customer contexts
            judge: Optional judge function

        Returns:
            Results dict with win rates
        """
        wins_a = 0
        wins_b = 0
        ties = 0

        for context in contexts:
            message_a = policy_a(context)
            message_b = policy_b(context)

            winner = self.compare_messages(message_a, message_b, context, judge)

            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1
            else:
                ties += 1

        total = len(contexts)
        results = {
            "policy_a_wins": wins_a,
            "policy_b_wins": wins_b,
            "ties": ties,
            "policy_a_win_rate": wins_a / total * 100,
            "policy_b_win_rate": wins_b / total * 100,
            "tie_rate": ties / total * 100,
        }

        return results

    def get_comparisons(self) -> List[Dict]:
        """Get all recorded comparisons."""
        return self.comparisons

