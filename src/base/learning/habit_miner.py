# base/learning/habit_miner.py
from __future__ import annotations

import datetime
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore

PROFILE_PATH = Path("config/user_profile.json")


def _time_bucket(ts: str) -> str:
    """Map ISO timestamp to a simple day-part bucket."""
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return "at unknown times"

    hour = dt.hour
    if 5 <= hour < 12:
        return "in the morning"
    elif 12 <= hour < 17:
        return "in the afternoon"
    elif 17 <= hour < 22:
        return "in the evening"
    else:
        return "late at night"


class HabitMiner:
    def __init__(self, db: SQLiteConn, memory, store: MemoryStore):
        self.db = db
        self.memory = memory
        self.habits: list[dict[str, Any]] = []  # cache
        self.store = store

    def load_profile(self) -> dict:
        if PROFILE_PATH.exists():
            try:
                return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load profile: {e}")
        return {}

    def save_profile(self, profile: dict) -> None:
        PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def learn(self, event: str, ts: str | None = None) -> None:
        """Record an event and try to identify recurring patterns."""
        timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
        self.habits.append({"action": event, "time": timestamp.time()})
        logger.debug(f"HabitMiner.learn: Added habit candidate {event} at {timestamp}")

    def extract_candidates(self, days: int = 30) -> list[tuple[str, str]]:
        """Pull recent events from memory with timestamps."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cur = self.store.conn.execute("SELECT content, ts FROM events WHERE ts >= ?", (cutoff,))
        return [(r["content"], r["ts"]) for r in cur.fetchall()]

    def summarize(self, candidates: list[tuple[str, str]], top_k: int = 5) -> list[str]:
        """
        Collapse repeated behaviors into short natural-language summaries,
        grouped by time of day.
        """
        # Count frequency by (content, bucket)
        counter = Counter()
        for content, ts in candidates:
            bucket = _time_bucket(ts)
            counter[(content, bucket)] += 1

        summaries = []
        for (content, bucket), count in counter.most_common(top_k):
            if count > 2:  # only habits, not one-offs
                summaries.append(f'User often says: "{content}" {bucket} (seen {count} times)')
        return summaries

    def get_summaries(self, days: int = 30, top_k: int = 5) -> list[str]:
        """Convenience wrapper to fetch recent events and return summaries."""
        candidates = self.extract_candidates(days=days)
        return self.summarize(candidates, top_k=top_k)

    def predict_next(self, action: str) -> datetime | None:
        """Very simple predictor: use the last recorded time for the action."""
        times = [h["time"] for h in self.habits if h["action"] == action]
        if not times:
            return None
        last_time = times[-1]
        now = datetime.utcnow()
        return datetime.combine(now.date(), last_time)

    def check_upcoming(self, minutes: int = 30) -> list[dict[str, Any]]:
        """
        Return habits likely to occur within the next `minutes`.
        Right now this is naive: looks at last seen time and compares to current time.
        """
        now = datetime.utcnow()
        upcoming = []
        for h in self.habits:
            scheduled = datetime.combine(now.date(), h["time"])
            delta = scheduled - now
            if timedelta(0) <= delta <= timedelta(minutes=minutes):
                upcoming.append(h)
        if upcoming:
            logger.info(f"HabitMiner.check_upcoming: {len(upcoming)} habit(s) due soon.")
        return upcoming

    def mine(self) -> dict:
        """
        Run enrichment based on habits and logs.
        Returns the updated profile (also saved to disk).
        """
        profile = self.load_profile()

        # ---- Mine favorite music habits ----
        cur = self.db.conn.execute("SELECT content, ts FROM events ORDER BY id DESC LIMIT 1000")
        rows = cur.fetchall()
        commands = []
        tod_hist = defaultdict(int)  # hour bucket histogram

        for r in rows:
            text = (r[0] or "").lower()
            ts = r[1] or ""
            try:
                hr = datetime.fromisoformat(ts).hour
            except Exception:
                hr = None
            if hr is not None:
                tod_hist[hr] += 1
            if text.startswith("play ") or "spotify" in text or "music" in text:
                commands.append("music")
            if "turn on light" in text or "lights" in text:
                commands.append("lights")
            if "calendar" in text or "schedule" in text or "meeting" in text:
                commands.append("calendar")

        if commands:
            c = Counter(commands)
            profile["most_used_commands"] = [k for k, _ in c.most_common(5)]

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

        # --- mine routines by time-of-day ---
        # store buckets like: morning/afternoon/evening/night
        buckets = {
            "morning": range(5, 12),
            "afternoon": range(12, 18),
            "evening": range(18, 23),
            "night": tuple(list(range(23, 24)) + list(range(0, 5))),
        }
        rout = {}
        if "music" in commands:
            # naïve example: if evenings dominate total events and music used recently, we assume evening music habit
            total_evening = sum(tod_hist[h] for h in buckets["evening"])
            if total_evening >= 10:
                rout["evening_music"] = "lo-fi"  # you can infer specific genre elsewhere

        if rout:
            profile["routines"] = rout

        self.save_profile(profile)
        return profile
        # # ---- Save changes ----
        # self.save_profile(profile)
        # return profile

    def reinforce(self, action: str) -> None:
        """
        Strengthen confidence in a habit/behavior and immediately enrich profile.
        """

        profile = self.load_profile()

        profile["reinforcements"] = profile.get("reinforcements", {})
        profile["reinforcements"][action] = profile["reinforcements"].get(action, 0) + 1
        self.save_profile(profile)
        self._update_persona_summary()

    def adjust(self, action: str) -> None:
        """
        Penalize or mark a behavior as needing correction and update profile.
        """

        profile = self.load_profile()

        profile["adjustments"] = profile.get("adjustments", {})
        profile["adjustments"][action] = profile["adjustments"].get(action, 0) + 1
        self.save_profile(profile)
        self._update_persona_summary()

    def _update_persona_summary(self) -> None:
        """
        Generate or refresh a natural-language persona block
        based on habits, reinforcements, and adjustments.
        """
        summary_lines = []
        profile = self.load_profile()

        if profile.get("reinforcements"):
            strong = sorted(profile["reinforcements"].items(), key=lambda kv: kv[1], reverse=True)
            for action, count in strong[:3]:
                summary_lines.append(f"Reinforced preference: {action} (x{count}).")

        if profile.get("adjustments"):
            weak = sorted(profile["adjustments"].items(), key=lambda kv: kv[1], reverse=True)
            for action, count in weak[:3]:
                summary_lines.append(f"Adjusted/avoided: {action} (x{count}).")

        # Save a running persona summary
        if summary_lines:
            profile["persona_summary"] = "\n".join(summary_lines)
            self.save_profile(profile)

    def export_summary(self) -> str:
        """
        Summarize mined habits/preferences into a natural-language block.
        Returns an empty string if nothing useful is stored.
        """

        habits = []
        profile = self.load_profile()
        lines = []

        # Example: frequent commands
        if profile.get("most_used_commands"):
            common = ", ".join(profile["most_used_commands"][:3])
            habits.append(f"Frequently uses commands like: {common}.")

        # Example: default preferences
        if profile.get("defaults"):
            for key, val in profile["defaults"].items():
                habits.append(f"Default {key}: {val}.")

        # Example: personality cues
        if profile.get("tone_bias"):
            habits.append(f"Tends to prefer replies that are {profile['tone_bias']}.")

        # # Add more here as HabitMiner grows
        # return "\n".join(habits) if habits else ""

        cmds = profile.get("most_used_commands") or []
        if cmds:
            lines.append("Frequently used domains: " + ", ".join(cmds) + ".")

        rout = profile.get("routines") or {}
        if rout.get("evening_music"):
            lines.append(f"In the evenings, user tends to play {rout['evening_music']} music.")

        # existing persona_summary (from reinforcement/adjust calls)
        if profile.get("persona_summary"):
            lines.append(profile["persona_summary"])

        # crisp preference flags
        if profile.get("tone_bias") == "succinct" or "Dislikes long answers" in profile.get(
            "persona_summary", ""
        ):
            lines.append("Dislikes long answers.")

        return "\n".join(lines)

    def prune_habits(self) -> None:
        """
        Nightly cleanup of reinforcement/adjustment data.
        - Decays counts
        - Removes weak habits
        - Consolidates strong ones into persona_summary
        """
        profile = self.load_profile()
        reinf = profile.get("reinforcements", {})
        adj = profile.get("adjustments", {})

        # Decay counts
        reinf = {k: v - 1 for k, v in reinf.items() if v > 1}
        adj = {k: v - 1 for k, v in adj.items() if v > 1}

        # Keep only strong habits
        reinf = {k: v for k, v in reinf.items() if v >= 3}
        adj = {k: v for k, v in adj.items() if v >= 3}

        profile["reinforcements"] = reinf
        profile["adjustments"] = adj

        # Refresh persona summary
        self._update_persona_summary()

    def _decay_factor(self, days_delta: float) -> float:
        return math.exp(-math.log(2) * (days_delta / 30.0))

    def update_from_usage(self) -> int:
        c = self.db.conn.cursor()
        c.execute(
            """
            WITH recent AS (
            SELECT id, resolved_action, params_json, created_at
            FROM usage_log ORDER BY id DESC LIMIT 5000
            )
            SELECT id, resolved_action, params_json, strftime('%s', 'now') - strftime('%s', created_at) AS age_sec
            FROM recent
            WHERE resolved_action IS NOT NULL
            """
        )
        updates = 0
        rows = c.fetchall()
        for r in rows:
            try:
                params = json.loads(r[2] or "{}")
            except Exception:
                params = {}
            age_days = max(0.0, (r[3] or 0) / 86400.0)
            df = self._decay_factor(age_days)

            keys = []
            if svc := params.get("service"):
                keys.append(f"music.service={svc}")
            if genre := params.get("genre"):
                keys.append(f"music.genre={genre}")
            if greet := params.get("greeting_style"):
                keys.append(f"ux.greeting_style={greet}")
            if bedtime := params.get("sleep_time"):
                keys.append(f"user.sleep_time={bedtime}")

            for key in keys:
                c.execute("SELECT id, count, score FROM habits WHERE key = ?", (key,))
                row = c.fetchone()
                if row:
                    hid, cnt, score = row
                    new_cnt = cnt + 1
                    new_score = score * 0.99 + df
                    c.execute(
                        "UPDATE habits SET count = ?, score = ?, last_used = CURRENT_TIMESTAMP WHERE id = ?",
                        (new_cnt, new_score, hid),
                    )
                else:
                    c.execute(
                        "INSERT INTO habits (key, count, score, last_used) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                        (key, 1, df),
                    )
                updates += 1

        self.db.conn.commit()
        logger.info(f"HabitMiner.update_from_usage: {updates} habit rows updated")
        return updates

    def top(self, prefix: str, n: int = 3):
        c = self.db.conn.cursor()
        c.execute(
            "SELECT key, count, score FROM habits WHERE key LIKE ? ORDER BY score DESC, count DESC LIMIT ?",
            (f"{prefix}%", n),
        )
        cols = ["key", "count", "score"]
        return [dict(zip(cols, r, strict=False)) for r in c.fetchall()]
