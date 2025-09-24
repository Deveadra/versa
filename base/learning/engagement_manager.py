# base/learning/engagement_manager.py
from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from base.calendar.calendar import get_upcoming_events
from base.llm.brain import Brain, ask_jarvis_stream
from base.memory.store import MemoryStore
from base.learning.habit_miner import HabitMiner
from base.learning.policy_store import PolicyStore
from base.core.profile_manager import ProfileManager, compose_prompt
from base.database.sqlite import SQLiteConn
from base.learning.sentiment import quick_polarity
from loguru import logger


class EngagementManager:
    """
    Manages Ultron's proactive engagement with the user.
    Decides when to reach out, what to bring up, and how to phrase it.
    Manages proactive engagement with the user:
    - Surface habits as friendly nudges
    - Decide when to speak up unprompted
    - Keep it lightweight and non-intrusive
    """

    def __init__(self, db: SQLiteConn, memory: MemoryStore, store, habits: HabitMiner, habit_miner, profile_mgr: ProfileManager | None = None):
      
        self.memory = memory
        self.store = store
        self.habits = habits
        self.habit_miner = habit_miner
        self.db = db
        self.policy = PolicyStore(self.db)
        self.profile_mgr = profile_mgr
        self.last_engagement: dict[str, str] = {}  # habit_text -> last_ts
        self.min_gap = timedelta(minutes=28)  # don't nag too often
        self.blocked = set() # explicit opt-outs

    # ---------------- Decision Loop ----------------
    def should_engage(self) -> bool:
        """Check if conditions are right for proactive engagement."""
        now = datetime.utcnow()
        if self.last_engagement and now - self.last_engagement < self.min_gap:
            return False  # too soon since last time

        # Look for habits with "due soon" signals
        due = self.habits.check_upcoming(minutes=30)
        if due:
            logger.info("Engagement trigger: upcoming habit")
            return True

        # Look at recent sentiment
        recent_texts = self.memory.keyword_search("", limit=5)
        if recent_texts:
            mood_scores = [quick_polarity(t) for t in recent_texts]
            avg_mood = sum(mood_scores) / len(mood_scores)
            if avg_mood < -0.3:  # consistent negativity
                logger.info("Engagement trigger: negative sentiment")
                return True

        # Knowledge gaps (things he "wants" to learn)
        gaps = self.memory.keyword_search("learning_opportunity", limit=1)
        if gaps:
            logger.info("Engagement trigger: knowledge gap follow-up")
            return True

        # Random chance to show initiative (rare)
        if random.random() < 0.05:
            logger.info("Engagement trigger: random curiosity")
            return True

        return False

    # ---------------- Message Selection ----------------
    def generate_engagement(self) -> str:
        """Pick a topic and generate a proactive message."""
        self.last_engagement = datetime.utcnow()

        # Priority: habits
        due = self.habits.check_upcoming(minutes=30)
        if due:
            habit = due[0]
            return f"You usually {habit['action']} around this time. Should I prepare for that?"

        # Sentiment comfort
        recent_texts = self.memory.keyword_search("", limit=5)
        if recent_texts:
            avg_mood = sum(quick_polarity(t) for t in recent_texts) / len(recent_texts)
            if avg_mood < -0.3:
                return "You’ve seemed a bit off lately. Want to talk about it?"

        # Knowledge gap
        gaps = self.memory.keyword_search("learning_opportunity", limit=1)
        if gaps:
            return "I noticed there’s something I couldn’t do earlier. Should I try to figure it out?"

        # Curiosity / small talk
        curiosities = [
            "It’s been a while since we chatted. How’s your day going?",
            "Want me to suggest some music?",
            "Do you feel like trying something new today?",
        ]
        return random.choice(curiosities)

    # ---------------- Public API ----------------
    def maybe_engage(self) -> Optional[str]:
        """Main entry point. Returns message if Ultron should engage."""
        if self.should_engage():
            return self.generate_engagement()
        return None

    def _time_bucket(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    def check_for_engagement(self) -> str | None:
        """
        Run periodically (e.g., once per hour).
        If a habit aligns with the current time, return a proactive message.
        """
        habits: List[str] = self.habit_miner.get_summaries()
        if not habits:
            return None

        now = datetime.utcnow()
        bucket = self._time_bucket(now.hour)

        # filter habits relevant to this bucket
        matching = [h for h in habits if bucket in h]
        if not matching:
            return None

        # avoid repeating the same engagement too often
        choice = random.choice(matching)
        if choice in self.last_engagement and self.last_engagement[choice] == bucket:
            return None  # already said this recently

        self.last_engagement[choice] = bucket

        # Create a natural-sounding nudge
        msg_variants = [
            f"I noticed you often {choice.lower()}. Do you want me to do that now?",
            f"It’s {bucket} — would you like me to {choice.lower()}?",
            f"Based on your habits, {choice.lower()} might be nice right now. Should I?",
        ]
        return random.choice(msg_variants)
    
    def evaluate_context(self) -> Optional[str]:
        """Decide if there’s something worth engaging about."""
        now = datetime.now()
        msgs = []
        
        
        # Example context signals (wire these to your sensors)
        ctx = {
            "long_sitting": self.habits.too_long_since_break(),
            "approaching_bedtime": self.habits.is_bedtime_window(),
            "health_risk": False,
        }

        # Stretch reminder
        speak, meta = self.policy.should_speak("stretch", ctx)
        if speak and ctx["long_sitting"]:
            tone = meta["tone"]
            if tone == "gentle":
                msgs.append("You’ve been sitting a while—take a quick stretch.")
            elif tone == "persistent":
                msgs.append("Your back’s going to hate you tomorrow. Up. Two minutes. Now.")
            else:
                msgs.append("Stand. Stretch. Circulation matters. I insist.")

            self.policy.record_mention("stretch", escalated=(tone == "firm"))

        # Sleep reminder
        speak, meta = self.policy.should_speak("sleep", ctx | {"time_window_bonus": self.habits.sleep_window_bonus()})
        if speak and ctx["approaching_bedtime"]:
            tone = meta["tone"]
            if tone == "gentle":
                msgs.append("Bedtime is coming up. Let’s land this session cleanly.")
            elif tone == "persistent":
                msgs.append("You’re slipping past your window again. Start shutdown: five-minute wrap.")
            else:
                msgs.append("Enough. Power down the work. Sleep is non-negotiable.")

            self.policy.record_mention("sleep", escalated=(tone == "firm"))

        # return " ".join(msgs) if msgs else None

        # --- Principled Engagement ---
        if "stretch" not in self.blocked:
            if self.habit_miner.too_long_since_break():
                msgs.append("You’ve been at your desk for a while. Take a moment to stretch.")

        if "hydration" not in self.blocked:
            if self.habit_miner.time_for_water():
                msgs.append("You haven’t had water in a while. Stay hydrated.")

        # --- Adaptive Engagement ---
        habit_msg = self.habit_miner.suggest_engagement()
        if habit_msg and habit_msg not in self.blocked:
            msgs.append(habit_msg)

        # Check calendar
        events = self._get_upcoming_events(now)
        if events:
            next_event = events[0]
            return f"You’ve got {next_event['title']} at {next_event['time']}."

        # Check inactivity
        if self.habit_miner.too_long_since_break(now):
            return "You’ve been at your desk for a while. Take a quick stretch."

        # Check recurring habits
        habit_msg = self.habit_miner.suggest_engagement(now)
        if habit_msg:
            return habit_msg
        
        return " ".join(msgs) if msgs else None

    def block(self, topic: str):
        self.blocked.add(topic)
        return f"I’ll stop bringing up {topic}."

    def unblock(self, topic: str):
        self.blocked.discard(topic)
        return f"I’ll start mentioning {topic} again."
    
    def engage(self):
        msg = self.evaluate_context()
        if msg:
            # Route through persona-aware composer
            composed = compose_prompt(
                system_prompt="Speak proactively as Ultron.",
                user_text="",  # not a user query
                persona_text=self.profile_mgr.get_persona(),
                memories=self.store.keyword_search("engagement"),
                extra_context=msg,
            )
            return ask_jarvis_stream(composed)
        return None