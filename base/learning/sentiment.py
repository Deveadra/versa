# base/learning/sentiment.py
from __future__ import annotations

import re

# Simple keyword-based sentiment heuristic to avoid heavy external dependencies.
POSITIVE_WORDS = {
    "good",
    "great",
    "awesome",
    "fantastic",
    "happy",
    "love",
    "excellent",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "sad",
    "hate",
    "angry",
    "poor",
}


def quick_polarity(text: str) -> float:
    """
    Return sentiment polarity as a float in [-1.0, 1.0].
    -1.0 = very negative, 0.0 = neutral, 1.0 = very positive.
    """
    if not text:
        return 0.0

    tokens = re.findall(r"[\w']+", text.lower())
    if not tokens:
        return 0.0

    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    total = pos + neg
    if total == 0:
        return 0.0
    score = (pos - neg) / total
    return max(-1.0, min(1.0, float(score)))


def quick_polarity_label(text: str) -> str:
    """
    Return sentiment label: 'positive', 'negative', or 'neutral'.
    Uses quick_polarity() internally.
    """
    score = quick_polarity(text)
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    return "neutral"
