# src/base/memory/scoring.py

from __future__ import annotations

import re


def assess_importance(text: str) -> int:
    """Cheap, tunable heuristic. Start conservative to avoid bloat."""
    score = 0
    t = text.lower()

    # Explicit cues
    if any(k in t for k in ("remember", "note this", "log this", "save this")):
        score += 50

    # Named-ish tokens (naive: capitalized mid-sentence)
    if re.search(r"\b[A-Z][a-z]+\b", text):
        score += 10

    # Numbers/dates
    if re.search(r"\d", text):
        score += 5

    # Length/structured info
    if len(text) >= 120:
        score += 5

    # Emotional cues (example)
    if any(k in t for k in ("happy", "angry", "upset", "excited", "worried")):
        score += 5

    return score
