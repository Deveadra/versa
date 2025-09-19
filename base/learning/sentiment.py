from __future__ import annotations

def quick_polarity(text: str) -> float:
    t = text.lower()
    pos = sum(t.count(w) for w in ["thanks", "great", "love", "nice", "perfect", "awesome"])
    neg = sum(t.count(w) for w in ["no", "not", "bad", "hate", "ugh", "terrible", "wrong"])
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(1.0, pos + neg)
