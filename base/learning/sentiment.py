from __future__ import annotations

from textblob import TextBlob  # or VADER, or your custom model

def quick_polarity(text: str) -> str:
    """
    Quick polarity detection: returns 'positive', 'negative', or 'neutral'.
    """
    if not text:
        return "neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

# def quick_polarity(text: str) -> float:
#     t = text.lower()
#     pos = sum(t.count(w) for w in ["thanks", "great", "love", "nice", "perfect", "awesome"])
#     neg = sum(t.count(w) for w in ["no", "not", "bad", "hate", "ugh", "terrible", "wrong"])
#     if pos == neg == 0:
#         return 0.0
#     return (pos - neg) / max(1.0, pos + neg)