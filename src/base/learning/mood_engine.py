# base/learning/mood_engine.py
from __future__ import annotations

from typing import Any

from loguru import logger

from base.database.sqlite import SQLiteConn
from base.learning.sentiment import quick_polarity
from base.policy.context_signals import ContextSignals


def _to_score(val: Any) -> float:
    """
    Normalize sentiment into a float in [-1.0, 1.0].
    Accepts either:
      - float/int (already numeric polarity)
      - 'positive' | 'neutral' | 'negative'
      - anything else → 0.0
    """
    if isinstance(val, (int, float)):
        x = float(val)
        return max(-1.0, min(1.0, x))
    s = str(val).strip().lower()
    if s == "positive":
        return 1.0
    if s == "negative":
        return -1.0
    if s == "neutral":
        return 0.0
    return 0.0


class MoodEngine:
    """
    Ultron’s mood tracker. Evaluates sentiment from user and assistant exchanges,
    persists mood history, and updates ContextSignals so policies can adapt.
    """

    def __init__(self, db_conn: SQLiteConn, context_signals: ContextSignals):
        self.db_conn = db_conn.conn if hasattr(db_conn, "conn") else db_conn
        self.context_signals = context_signals
        self.mood: str = "neutral"
        self.mood_score: float = 0.0  # -1.0 (very negative) → +1.0 (very positive)
        self.mood_history: list[dict[str, Any]] = []

        self.load_mood_history()
        self.evaluate_mood()

    # ----------------- Persistence -----------------

    def load_mood_history(self):
        cur = self.db_conn.cursor()
        cur.execute("SELECT * FROM mood ORDER BY timestamp ASC")
        rows = cur.fetchall()
        cur.close()

        self.mood_history = [dict(r) for r in rows] if rows else []
        if self.mood_history:
            last = self.mood_history[-1]
            self.mood = last.get("mood", "neutral")
            self.mood_score = float(last.get("score", 0.0))
        logger.debug(
            f"Loaded {len(self.mood_history)} mood records (current={self.mood}, score={self.mood_score})"
        )

    def save_mood_entry(self):
        cur = self.db_conn.cursor()
        cur.execute(
            "INSERT INTO mood (timestamp, mood, score) VALUES (CURRENT_TIMESTAMP, ?, ?)",
            (self.mood, self.mood_score),
        )
        self.db_conn.commit()
        cur.close()
        logger.debug(f"Saved mood entry: {self.mood} ({self.mood_score})")

    # ----------------- Evaluation -----------------
    def evaluate_mood(self):
        """
        Push the current mood into ContextSignals so the policy engine can use it.
        """
        self.context_signals.upsert("mood_score", round(self.mood_score, 3), type_="counter")
        self.context_signals.upsert("mood", self.mood, type_="string")  # or "text"

        flags = [
            "happy",
            "frustrated",
            "neutral",
            "sad",
            "anxious",
            "excited",
            "bored",
            "stressed",
            "calm",
            "focused",
            "distracted",
            "productive",
            "unproductive",
            "confident",
            "insecure",
        ]
        for f in flags:
            if self.mood == f:
                self.context_signals.mark_true(f"is_{f}")
            else:
                self.context_signals.mark_false(f"is_{f}")

        logger.debug(f"Evaluated mood → {self.mood} ({self.mood_score})")

    # ----------------- Updates -----------------
    def update_mood(self, user_input: str, assistant_response: str):
        """
        Update mood score based on user + assistant sentiment.
        Works whether quick_polarity returns a string label or a numeric score.
        """
        user_raw = quick_polarity(user_input)
        asst_raw = quick_polarity(assistant_response)

        user_score = _to_score(user_raw)
        asst_score = _to_score(asst_raw)
        combined = (user_score + asst_score) / 2.0

        # exponential moving average for stability
        self.mood_score = max(-1.0, min(1.0, self.mood_score * 0.8 + combined * 0.2))

        if self.mood_score > 0.3:
            self.mood = "happy"
        elif self.mood_score < -0.3:
            self.mood = "frustrated"
        else:
            self.mood = "neutral"

        logger.debug(f"Updated mood → {self.mood} ({self.mood_score:.3f})")
        self.save_mood_entry()
        self.evaluate_mood()

    def get_current_mood(self) -> str:
        """Return current mood string (defaults to neutral)."""
        return self.mood or "neutral"
