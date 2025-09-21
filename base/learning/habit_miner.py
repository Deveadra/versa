
# base/learning/habit_miner.py
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
from loguru import logger

from base.database.sqlite import SQLiteConn

PROFILE_PATH = Path("config/user_profile.json")


class HabitMiner:
    def __init__(self, db: SQLiteConn):
        self.db = db

    def load_profile(self) -> dict:
        if PROFILE_PATH.exists():
            try:
                return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load profile: {e}")
        return {}

    def save_profile(self, profile: dict) -> None:
        PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def mine(self) -> dict:
        """
        Run enrichment based on habits and logs.
        Returns the updated profile (also saved to disk).
        """
        profile = self.load_profile()

        # ---- Mine favorite music habits ----
        cur = self.db.conn.execute(
            "SELECT content FROM events WHERE content LIKE '%play %' ORDER BY id DESC LIMIT 200"
        )
        phrases = [r[0] for r in cur.fetchall()]
        if phrases:
            cleaned = [p.lower().replace("play", "").strip() for p in phrases]
            common = Counter(cleaned)
            top, freq = common.most_common(1)[0]
            if freq >= 3:  # only save if habit is repeated
                profile["favorite_music"] = top
                logger.info(f"HabitMiner: favorite_music → {top}")

        # ---- Adjust tone preferences ----
        tone_adj = profile.get("tone_adjustments", {"casual": 0.5, "formal": 0.5})
        for phrase in phrases:
            if "sarcasm" in phrase.lower() or "joke" in phrase.lower():
                tone_adj["playful"] = min(tone_adj.get("playful", 0.0) + 0.05, 1.0)
        profile["tone_adjustments"] = tone_adj

        # ---- Greeting refinement ----
        cur = self.db.conn.execute(
            "SELECT content FROM events WHERE content LIKE '%hello%' OR content LIKE '%hi %'"
        )
        greetings = [r[0] for r in cur.fetchall()]
        if greetings:
            common = Counter(greetings)
            top, freq = common.most_common(1)[0]
            if freq >= 2:
                profile["preferred_greeting"] = top.strip()
                logger.info(f"HabitMiner: preferred_greeting → {top.strip()}")

        # ---- Save changes ----
        self.save_profile(profile)
        return profile
