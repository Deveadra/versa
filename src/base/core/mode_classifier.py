import random
import re

# Global memory (lives until process ends)
_last_input = None
_repeat_count = 0


def classify_mode(user_text: str, last_mode: str = "default") -> tuple[str, bool]:
    """
    Returns (mode, repeat_triggered) tuple.
    """
    """
    Heuristic classifier to choose between default, sarcastic, or formal.
    Uses probabilistic triggers and remembers repeats across the session.
    """
    global _last_input, _repeat_count

    text = user_text.lower().strip()
    repeat_triggered = False

    if _last_input == text:
        _repeat_count += 1
    else:
        _repeat_count = 0
        _last_input = text

    # Formal
    if any(word in text for word in ["email", "compose", "calendar", "meeting"]):
        return "formal", False

    # Sarcasm triggers
    if any(word in text for word in ["duh", "obvious", "really", "seriously", "bruh"]) or re.search(
        r"\bwhat is 2\s*\+\s*2\b", text
    ):
        if random.random() < 0.7:
            return "sarcastic", False

    # Repeat escalation
    if _repeat_count >= 2:
        if random.random() < 0.9:
            repeat_triggered = True
            return "sarcastic", repeat_triggered

    return "default", False
