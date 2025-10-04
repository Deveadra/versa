# base/core/decider.py
import hashlib
import re

from datetime import datetime

from base.core.nlu import parse_diagnostic_intent

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
    "low_value": -50,
}

# Example regex checks (lightweight)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/\d{1,2})\b", re.I
)

# Simple phrase checks
REMEMBER_PHRASES = ["remember", "don't forget", "note that", "remind me that", "save this"]
PREFERENCE_PHRASES = [
    "i like",
    "i love",
    "i dislike",
    "i hate",
    "my favorite",
    "i prefer",
    "i don't like",
]


class Decider:
    def __init__(self):
        # track repeats in-session
        self.session_repeat_counter = {}

    def _has_named_entity(self, text: str) -> bool:
        return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or DATE_RE.search(text))

    def _is_direct_command(self, text: str) -> bool:
        # e.g., "set my timezone to X", "my name is X", "call me X"
        patterns = [r"\bmy name is\b", r"\bcall me\b", r"\bset my\b", r"\bchange my\b"]
        return any(re.search(p, text, re.I) for p in patterns)

    def _is_preference(self, text: str) -> bool:
        return any(p in text.lower() for p in PREFERENCE_PHRASES)

    def _is_explicit_remember(self, text: str) -> bool:
        return any(p in text.lower() for p in REMEMBER_PHRASES)
    
    def score_text(
        self, text: str, plugin_hint: str = "", repeat_count: int = 0
    ) -> tuple[int, dict]:
        """
        Compute importance score and suggested memory metadata.
        """
        intent = parse_diagnostic_intent(text)
        score = 0
        meta = {"category": None, "reason": []}

        if self._is_explicit_remember(text):
            score += WEIGHTS["explicit_remember"]
            meta["reason"].append("explicit_remember")

        if self._is_direct_command(text):
            score += WEIGHTS["direct_command"]
            meta["reason"].append("direct_command")

        if self._has_named_entity(text):
            score += WEIGHTS["named_entity"]
            meta["reason"].append("named_entity")

        if self._is_preference(text):
            score += WEIGHTS["preference"]
            meta["reason"].append("preference")

        if len(text) > 100:
            score += WEIGHTS["length"]
            meta["reason"].append("long_text")

        if repeat_count > 0:
            score += WEIGHTS["repetition"] * repeat_count
            meta["reason"].append(f"repeat_{repeat_count}")

        if plugin_hint:
            score += WEIGHTS["plugin_priority"]
            meta["reason"].append(f"plugin_hint:{plugin_hint}")

        low_value_markers = [
            "how are you",
            "what's up",
            "hello",
            "hi",
            "thanks",
            "thank you",
            "bye",
        ]
        
        if "diagnostic" in text.lower() or "scan" in text.lower():
            score += WEIGHTS["direct_command"]
            meta["reason"].append("diagnostic_command")
            meta["category"] = "action"

        if intent:
            # Prefer WEIGHTS if present, else use a sane bump
            bump = WEIGHTS["direct_command"] if "WEIGHTS" in globals() and "direct_command" in WEIGHTS else 2.0
            score += bump
            # Ensure meta has the fields you use elsewhere
            meta.setdefault("reason", []).append("diagnostic_command")
            meta["category"] = "action"
            meta["intent"] = "diagnostic"
            meta["intent_payload"] = {"mode": intent["mode"], "fix": intent["fix"], "confidence": intent["confidence"]}
            
        if any(m in text.lower() for m in low_value_markers):
            score += WEIGHTS["low_value"]
            meta["reason"].append("low_value_marker")

        if self._is_preference(text):
            meta["category"] = "preference"
        elif self._is_direct_command(text) or self._has_named_entity(text):
            meta["category"] = "fact"
        elif plugin_hint == "calendar" or "meeting" in text.lower():
            meta["category"] = "event"
        else:
            meta["category"] = "short_history"

        return score, meta
    
    def decide(self, text: str, plugin_hint: str = "") -> tuple[int, dict]:
        """
        Wrapper around score_text that keeps repeat counts.
        Returns (score, metadata).
        """
        norm = text.strip().lower()
        self.session_repeat_counter[norm] = self.session_repeat_counter.get(norm, 0) + 1
        repeat_count = self.session_repeat_counter[norm] - 1
        return self.score_text(text, plugin_hint, repeat_count)

    def extract_structured_fact(self, text: str):
        # very small heuristics; expand later
        m = re.search(r"my name is ([A-Za-z\s'-]+)", text, re.I)
        if m:
            return ("user_name", m.group(1).strip())
        m2 = re.search(r"i live in ([A-Za-z\s]+)", text, re.I)
        if m2:
            return ("location", m2.group(1).strip())
        return None

    def decide_memory(self, user_text: str, reply: str) -> dict | None:
        """
        Decide if the exchange is worth remembering.
        Returns a dict if yes, None if no.
        """
        important_keywords = [
            "my name is",
            "remember",
            "birthday",
            "favorite",
            "I like",
            "I don’t like",
        ]

        if any(kw in user_text.lower() for kw in important_keywords):
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "fact",
                "content": user_text,
                "response": reply,
            }

        if "I will remember" in reply or "Noted" in reply:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "acknowledgement",
                "content": user_text,
                "response": reply,
            }

        return None

    def handle_user_text(self, text: str, plugin_hint: str = None or ""):
        norm = text.strip().lower()
        self.session_repeat_counter[norm] = self.session_repeat_counter.get(norm, 0) + 1

        score, meta = self.score_text(
            text,
            plugin_hint=(plugin_hint or ""),
            repeat_count=self.session_repeat_counter[norm] - 1,
        )

        STORE_THRESHOLD = 60  # tuneable
        if score >= STORE_THRESHOLD or "explicit_remember" in meta.get("reason", []):
            key = self.make_fact_key(text, hint=(meta.get("category") or ""))
            existing = memory.recall_fact(key)
            if existing is None:
                value = text.strip()
                memory.remember_fact(key, value)
            else:
                memory.remember_fact(key, existing)  # refresh timestamp

        memory.add_history(text)

    

    def make_fact_key(self, text: str, hint: str = "") -> str:
        base = (hint or "") + "|" + text.strip().lower()
        return hashlib.sha256(base.encode("utf-8")).hexdigest()
