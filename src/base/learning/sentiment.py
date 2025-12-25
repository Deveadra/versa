# base/learning/sentiment.py
from __future__ import annotations

from typing import Any

from textblob import TextBlob  # type: ignore


def quick_polarity(text: str) -> float:
    """
    Return sentiment polarity as a float in [-1.0, 1.0].
    -1.0 = very negative, 0.0 = neutral, 1.0 = very positive.
    """
    if not text:
        return 0.0
    blob: Any = TextBlob(text)
    sentiment: Any = getattr(blob, "sentiment", None)
    polarity = getattr(sentiment, "polarity", 0.0)
    try:
        return float(polarity)
    except Exception:
        return 0.0


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
