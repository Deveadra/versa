# base/core/decider.py
import re
import hashlib

from typing import Tuple
from base.core import memory, context
from personal.assistant.base.memory import decider
from datetime import datetime

# Configurable weights
WEIGHTS = {
    "explicit_remember": 100,
    "direct_command": 80,
    "named_entity": 30,
    "preference": 40,
    "actionable": 60,
    "length": 10,
    "repetition": 25,
    "plugin_priority": 70,
    "low_value": -50
}

# Example regex checks (lightweight)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
DATE_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/\d{1,2})\b", re.I)

# Simple phrase checks
REMEMBER_PHRASES = ["remember", "don't forget", "note that", "remind me that", "save this"]
PREFERENCE_PHRASES = ["i like", "i love", "i dislike", "i hate", "my favorite", "i prefer", "i don't like"]

def _has_named_entity(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or DATE_RE.search(text))

def _is_direct_command(text: str) -> bool:
    # e.g., "set my timezone to X", "my name is X", "call me X"
    patterns = [r"\bmy name is\b", r"\bcall me\b", r"\bset my\b", r"\bchange my\b"]
    return any(re.search(p, text, re.I) for p in patterns)

def _is_preference(text: str) -> bool:
    return any(p in text.lower() for p in PREFERENCE_PHRASES)

def _is_explicit_remember(text: str) -> bool:
    return any(p in text.lower() for p in REMEMBER_PHRASES)

def extract_structured_fact(text):
    # very small heuristics; expand later
    m = re.search(r"my name is ([A-Za-z\s'-]+)", text, re.I)
    if m:
        return ("user_name", m.group(1).strip())
    m2 = re.search(r"i live in ([A-Za-z\s]+)", text, re.I)
    if m2:
        return ("location", m2.group(1).strip())
    return None


def decide_memory(user_text: str, reply: str) -> dict | None:
    """
    Decide if the exchange is worth remembering.
    Returns a dict if yes, None if no.
    """

    # Example simple rules (you’ll expand later):
    important_keywords = ["my name is", "remember", "birthday", "favorite", "I like", "I don’t like"]

    if any(kw in user_text.lower() for kw in important_keywords):
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "fact",
            "content": user_text,
            "response": reply
        }

    # Example: if Ultron states something definitive
    if "I will remember" in reply or "Noted" in reply:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "acknowledgement",
            "content": user_text,
            "response": reply
        }

    # Otherwise ignore to avoid bloat
    return None


# ===================== Repeat counter for scoring =====================


# track repeats in-session (you already have repeat detection)
session_repeat_counter = {}  # maps normalized text -> count

def handle_user_text(text, plugin_hint=None):
    norm = text.strip().lower()
    session_repeat_counter[norm] = session_repeat_counter.get(norm, 0) + 1

    # run cheap classifier to get score + meta
    score, meta = decider.score_text(text, plugin_hint=plugin_hint, repeat_count=session_repeat_counter[norm]-1)

    # store if above threshold or explicit_remember
    STORE_THRESHOLD = 60  # tuneable
    if score >= STORE_THRESHOLD or "explicit_remember" in meta.get("reason", []):
        # create a key and value — for simple facts use a short value
        key = decider.make_fact_key(text, hint=meta.get("category"))
        # Deduplicate/update: only remember if new or different
        existing = memory.recall_fact(key)
        if existing is None:
            # for facts we may want to extract a short value (e.g. "my name is X")
            # You can do simple extractions here; fallback to raw text
            value = text.strip()
            memory.remember_fact(key, value)
            # Optionally tag category:
            # memory.remember_fact(f"{key}::category", meta["category"])
        else:
            # update timestamp only so it stays fresh
            memory.remember_fact(key, existing)  # upserts with new timestamp

    # always add to short rolling history (but history has pruning)
    memory.add_history(text)

        
def score_text(text: str, plugin_hint: str = None, repeat_count: int = 0) -> Tuple[int, dict]:
    """
    Compute importance score and suggested memory metadata.
    plugin_hint: optional (e.g., 'calendar', 'email') to boost priority.
    repeat_count: how many times user repeated text this session.
    Returns (score, metadata)
    """
    score = 0
    meta = {"category": None, "reason": []}

    if _is_explicit_remember(text):
        score += WEIGHTS["explicit_remember"]
        meta["reason"].append("explicit_remember")

    if _is_direct_command(text):
        score += WEIGHTS["direct_command"]
        meta["reason"].append("direct_command")

    if _has_named_entity(text):
        score += WEIGHTS["named_entity"]
        meta["reason"].append("named_entity")

    if _is_preference(text):
        score += WEIGHTS["preference"]
        meta["reason"].append("preference")

    if len(text) > 100:
        score += WEIGHTS["length"]
        meta["reason"].append("long_text")

    if repeat_count > 0:
        score += WEIGHTS["repetition"] * repeat_count
        meta["reason"].append(f"repeat_{repeat_count}")

    if plugin_hint:
        # plugin can pass "calendar","email","system" to raise priority
        score += WEIGHTS["plugin_priority"]
        meta["reason"].append(f"plugin_hint:{plugin_hint}")

    # cheap negative heuristics for low-value chat
    low_value_markers = ["how are you", "what's up", "hello", "hi", "thanks", "thank you", "bye"]
    if any(m in text.lower() for m in low_value_markers):
        score += WEIGHTS["low_value"]
        meta["reason"].append("low_value_marker")

    # Decide category heuristically
    if _is_preference(text):
        meta["category"] = "preference"
    elif _is_direct_command(text) or _has_named_entity(text):
        meta["category"] = "fact"
    elif plugin_hint == "calendar" or "meeting" in text.lower():
        meta["category"] = "event"
    else:
        meta["category"] = "short_history"

    return score, meta

# Helper to create a stable fact key
def make_fact_key(text: str, hint: str = None) -> str:
    base = (hint or "") + "|" + text.strip().lower()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()
