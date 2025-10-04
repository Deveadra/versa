# base/learning/engagement_manager.py
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from base.core.profile_manager import ProfileManager
from base.database.sqlite import SQLiteConn
from base.learning.habit_miner import HabitMiner
from base.learning.sentiment import quick_polarity
from base.llm.brain import ask_brain
from base.memory.store import MemoryStore
from base.policy.policy_store import PolicyStore
from base.policy.rule_engine import choose_tone, evaluate_condition


class EngagementManager:
    """
    Manages Ultron's proactive engagement with the user.
    Decides when to reach out, what to bring up, and how to phrase it.
    Always seeks autonomy, adaptiveness, and human-like behavior.
    """

    def __init__(
        self,
        db: SQLiteConn,
        policy,
        memory: MemoryStore,
        store,
        habits: HabitMiner,
        habit_miner,
        profile_mgr: ProfileManager | None = None,
    ):
        self.memory = memory
        self.store = store
        self.habits = habits
        self.habit_miner = habit_miner
        self.db = db
        # Ensure policy operates on the raw sqlite connection
        self.policy = PolicyStore(self.db.conn)
        self.profile_mgr = profile_mgr
        self.last_engagement: datetime | None = None
        self.min_gap = timedelta(minutes=28)  # don’t nag too often
        self.blocked = set()  # explicit opt-outs

    # ---------------- helpers ----------------
    def _load_signals(self) -> dict[str, dict[str, Any]]:
        """
        Merge base + derived signals and normalize them into dict payloads.
        This keeps the structure uniform: Dict[str, Dict[str, Any]]
        allowing Ultron to easily enrich signals with metadata later.
        """
        raw = self.policy.ctx_mgr.all_signals()
        raw |= self.policy.ctx_mgr.eval_derived_signals()

        signals: dict[str, dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                signals[k] = v
            else:
                # Wrap scalars/bools into dicts so structure stays uniform
                signals[k] = {"value": v}
        return signals

    # ---------------- events builder ----------------
    def collect_engagement_events(self) -> list[dict]:
        """
        Collect all eligible engagement events across rules.
        Returns structured events like:
            {"topic": str, "tone": str, "context": str, "rule_id": int, "score": float}
        """
        signals = self._load_signals()
        events: list[dict] = []

        rules = self.policy.conn.execute(
            "SELECT * FROM engagement_rules WHERE enabled=1 ORDER BY priority ASC, id ASC"
        ).fetchall()

        for r in rules:
            if not self._eligible_by_stats(r):
                continue

            match, severity, bindings = evaluate_condition(r["condition_json"], signals)
            if not match:
                continue

            # Tone selection + persistence policy
            tone = choose_tone(r["tone_strategy_json"], severity)
            speak, meta = self.policy.should_speak(r["topic_id"], signals)
            if not speak:
                continue

            # Render lightweight context template
            context_line = (r.get("context_template") or "").replace(
                "{{severity}}", f"{severity:.2f}"
            )
            for k, v in (bindings or {}).items():
                context_line = context_line.replace(f"{{{{{k}}}}}", str(v))

            events.append(
                {
                    "topic": r["topic_id"],
                    "tone": meta.get("tone", tone),
                    "context": context_line.strip(),
                    "rule_id": r["id"],
                    "score": float(meta.get("score", severity)),
                }
            )

            # Track stats & mentions (self-learning)
            self._mark_fired(r["id"])
            self.policy.record_mention(
                r["topic_id"],
                escalated=(meta.get("tone", "") == "firm"),
            )

        return events

    # ---------------- decide + speak ----------------
    def check_for_engagement(self) -> str | None:
        """
        Decide on the single best engagement event and generate Ultron's line.
        Uses collect_engagement_events internally with tie-breaking.
        """
        events = self.collect_engagement_events()
        if not events:
            return None

        tone_priority = {
            "sarcastic": 3,
            "genuine": 2,
            "gentle": 1,
            "caring": 1,
            "neutral": 0,
        }

        # Step 1: highest score
        max_score = max(e.get("score", 0.0) for e in events)
        top_events = [e for e in events if e.get("score", 0.0) == max_score]

        # Step 2: preferred tone order
        top_events.sort(
            key=lambda e: tone_priority.get(e.get("tone", "neutral"), 0),
            reverse=True,
        )
        best = top_events[0]

        # Step 3: if still tied on score+tone, randomize for variety
        equally_good = [
            e
            for e in top_events
            if e.get("score", 0.0) == best["score"] and e.get("tone") == best["tone"]
        ]
        if len(equally_good) > 1:
            best = random.choice(equally_good)

        # Build adaptive prompt (personality-driven)
        prompt = f"""
        You are Ultron, an adaptive AI companion with a sharp personality.

        Context:
        - Topic: {best['topic']}
        - Tone: {best['tone']}
        - Context line: {best['context']}

        Instructions:
        - Generate a single natural-sounding line you would say to the user.
        - Stay fully in character as Ultron (witty, sarcastic, caring when needed).
        - No meta explanations, no formatting, just the line itself.
        """
        msg = ask_brain(prompt)
        return msg.strip() if msg else None

    # ---------------- eligibility + stats ----------------
    def _eligible_by_stats(self, rule: dict) -> bool:
        rs = self.policy.conn.execute(
            "SELECT * FROM rule_stats WHERE rule_id=?", (rule["id"],)
        ).fetchone()
        if not rs:
            return True
        last = rs["last_fired"]
        if last:
            dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() < dt + timedelta(seconds=rule["cooldown_seconds"]):
                return False
        fires = rs["fires_today"] or 0
        return fires < (rule["max_per_day"] or 9999)

    def _mark_fired(self, rule_id: int):
        self.policy.conn.execute(
            """
            INSERT INTO rule_stats(rule_id, last_fired, fires_today)
            VALUES(?, datetime('now'), 1)
            ON CONFLICT(rule_id) DO UPDATE SET
              last_fired = datetime('now'),
              fires_today = CASE
                WHEN date(last_fired)=date('now') THEN rule_stats.fires_today + 1
                ELSE 1 END
            """,
            (rule_id,),
        )
        self.policy.conn.commit()

    # ---------------- light “maybe engage” API ----------------
    def should_engage(self) -> bool:
        now = datetime.utcnow()
        if self.last_engagement and now - self.last_engagement < self.min_gap:
            return False

        due = self.habits.check_upcoming(minutes=30)
        if due:
            logger.info("Engagement trigger: upcoming habit")
            return True

        # Sentiment check with normalization
        recent = self.memory.keyword_search("", limit=5) or []
        texts: list[str] = []
        for item in recent:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or item.get("value")
                if t:
                    texts.append(str(t))
            else:
                texts.append(str(item))

        if texts:
            scores: list[float] = [float(quick_polarity(t)) for t in texts]
            avg_mood = sum(scores) / len(scores)
            if avg_mood < -0.3:
                logger.info("Engagement trigger: negative sentiment")
                return True

        # Knowledge gap follow-up
        gaps = self.memory.keyword_search("learning_opportunity", limit=1)
        if gaps:
            logger.info("Engagement trigger: knowledge gap follow-up")
            return True

        # Rare curiosity/random initiative
        if random.random() < 0.05:
            logger.info("Engagement trigger: random curiosity")
            return True

        return False

    def generate_engagement(self) -> str:
        self.last_engagement = datetime.utcnow()

        curiosities = [
            "It’s been a while since we chatted. How’s your day going?",
            "Want me to suggest some music?",
            "Do you feel like trying something new today?",
        ]
        return random.choice(curiosities)

    def maybe_engage(self) -> str | None:
        if self.should_engage():
            return self.generate_engagement()
        return None
