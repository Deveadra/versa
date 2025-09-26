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
from base.policy.rule_engine import evaluate_condition, choose_tone
from base.database.sqlite import SQLiteConn
from base.learning.sentiment import quick_polarity
from base.policy.rule_engine import derive_expectation
from base.policy.tone_memory import get_tone
from base.policy.tone_memory import choose_tone_for_topic
from base.policy.consequence_linker import detect_consequence
from base.policy.consequence_linker import style_complaint
from base.policy.tone_memory import choose_tone_for_topic
from base.llm.brain import ask_jarvis_stream


from loguru import logger


signals, expect_fn = derive_expectation(r["condition_json"])
watch_signal, expect_fn = (sig_expect if sig_expect else (None, None))
tone = choose_tone(r["tone_strategy_json"], severity)
# Pick tone adaptively using tone memory
tone = choose_tone_for_topic(self.db, rule["topic_id"])
tone = get_tone(self.policy.conn, r["topic_id"], tone)

events.append({
    "topic": r["topic_id"],
    "tone": meta.get("tone", tone),
    "context": context.strip(),
    "rule_id": r["id"],
    "score": meta.get("score", severity),
    "watch_signals": signals,
    "expect_change": expect_fn,
})


class EngagementManager:
    """
    Manages Ultron's proactive engagement with the user.
    Decides when to reach out, what to bring up, and how to phrase it.
    Manages proactive engagement with the user:
    - Surface habits as friendly nudges
    - Decide when to speak up unprompted
    - Keep it lightweight and non-intrusive
    """

    def __init__(self, db: SQLiteConn, policy, memory: MemoryStore, store, habits: HabitMiner, habit_miner, profile_mgr: ProfileManager | None = None):
      
        self.memory = memory
        self.store = store
        self.habits = habits
        self.habit_miner = habit_miner
        self.db = db
        self.policy = policy
        self.policy = PolicyStore(self.db)
        self.profile_mgr = profile_mgr
        self.last_engagement: dict[str, str] = {}  # habit_text -> last_ts
        self.min_gap = timedelta(minutes=28)  # don't nag too often
        self.blocked = set() # explicit opt-outs

    # ---------------- Decision Loop ----------------
    def _load_signals(self) -> Dict[str,dict]:
        # Providers update signals elsewhere; here we just read all of them.
        # (If you want HabitMiner to contribute live values, write them into ContextManager first.)
        ctx = self.policy.ctx_mgr.all_signals()
        ctx |= self.policy.ctx_mgr.eval_derived_signals()
        return ctx

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
        # Max per day check
        fires = rs["fires_today"] or 0
        return fires < (rule["max_per_day"] or 9999)

    def _mark_fired(self, rule_id: int):
        # upsert stats
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

    def check_for_engagement(self) -> List[dict]:
        """
        Returns a list of structured engagement events:
        [{"topic": ..., "tone": ..., "context": ..., "rule_id": ..., "score": ...}, ...]
        """
        signals = self._load_signals()
        events: List[dict] = []

        rules = self.policy.conn.execute(
            "SELECT * FROM engagement_rules WHERE enabled=1 ORDER BY priority ASC, id ASC"
        ).fetchall()
        cur = self.db.cursor()
        row = cur.execute("""
            SELECT * FROM engagement_rules
            WHERE enabled=1
            ORDER BY priority DESC
            LIMIT 1
        """).fetchone()

        if not row:
            return None

        # Adaptive tone from memory
        # Build adaptive message
        tone = choose_tone_for_topic(self.db, row["topic_id"])

        # Fetch cluster + last complaint
        cluster_row = cur.execute("""
            SELECT cluster, last_example FROM complaint_clusters
            WHERE topic_id=?
            ORDER BY last_updated DESC
            LIMIT 1
        """, (row["topic_id"],)).fetchone()

        cluster = cluster_row["cluster"] if cluster_row else None
        last_example = style_complaint(cluster_row["last_example"]) if cluster_row and cluster_row["last_example"] else None
        cluster_row = cur.execute("""
            SELECT cluster, last_example FROM complaint_clusters
            WHERE topic_id=?
            ORDER BY last_updated DESC
            LIMIT 1
        """, (row["topic_id"],)).fetchone()

        cluster = cluster_row["cluster"] if cluster_row else None
        


        # Build adaptive message
        if tone == "genuine":
            if cluster and last_example:
                msg = f"Time to address {row['topic_id']} — remember when you said '{last_example}'?"
            elif cluster:
                msg = f"Time to address {row['topic_id']} — all those {cluster} aren’t random."
            else:
                msg = f"Time to pay attention to {row['topic_id']}."
        elif tone == "sarcastic":
            if last_example:
                msg = f"Oh, ignore me about {row['topic_id']} again. But don’t come whining with '{last_example}' later."
            elif cluster:
                msg = f"Oh sure, ignore me about {row['topic_id']}. Worked great with all your {cluster}, didn’t it?"
            else:
                msg = f"Funny how you keep ignoring me about {row['topic_id']}."
        else:
            msg = f"Reminder: {row['topic_id']}."

        return msg

            # 3) Build context payload for the brain
        context = {
            "topic": row["topic_id"],
            "tone": tone,
            "cluster": cluster,
            "last_complaint": last_example,
        }

        # 4) Generate natural line via LLM
        prompt = f"""
        You are Ultron, an adaptive AI companion with a sharp personality.
        The user may ignore your advice often. Use the provided context to generate a
        single natural-sounding line you would say to the user.

        Context:
        - Topic: {context['topic']}
        - Tone: {context['tone']}
        - Cluster: {context['cluster']}
        - Last complaint: {context['last_complaint']}

        Constraints:
        - Only generate the line (no explanations, no meta-commentary).
        - Tone should reflect frustration, sarcasm, or care as instructed.
        - If there’s a last complaint, you may quote or paraphrase it.
        - Stay in character as Ultron.
        """

        msg = ask_jarvis_stream(prompt)
        return msg.strip() if msg else None
       continue
    
        # Adaptive tone from memory        

        for r in rules:
            if not self._eligible_by_stats(r):
                continue

            match, severity, bindings = evaluate_condition(r["condition_json"], signals)
            if not match:
                continue

            tone = choose_tone(r["tone_strategy_json"], severity)

            # PolicyStore still governs principled/advocate/adaptive persistence per topic
            speak, meta = self.policy.should_speak(r["topic_id"], signals)
            if not speak:
                continue

            # Render context template (lightweight, no Jinja)
            context = (r["context_template"] or "") \
                .replace("{{severity}}", f"{severity:.2f}")
            for k, v in (bindings or {}).items():
                context = context.replace(f"{{{{{k}}}}}", str(v))

            events.append({
                "topic": r["topic_id"],
                "tone": meta.get("tone", tone),
                "context": context.strip(),
                "rule_id": r["id"],
                "score": meta.get("score", severity),
            })

            self._mark_fired(r["id"])
            self.policy.record_mention(r["topic_id"], escalated=(meta.get("tone","") == "firm"))

        return events
    
    def load_context(self) -> dict:
        # Load all base + derived signals
        ctx = self.policy.ctx_mgr.all_signals()
        ctx |= self.policy.ctx_mgr.eval_derived_signals()

        # HabitMiner signals get pushed into ContextManager
        self.policy.ctx_mgr.set_signal("long_sitting", self.habits.too_long_since_break(), source="habit_miner")
        self.policy.ctx_mgr.set_signal("approaching_bedtime", self.habits.is_bedtime_window(), source="habit_miner")

        return {**ctx, **self.policy.ctx_mgr.eval_derived_signals()}


    def evaluate_context(self) -> Optional[str]:
        """Decide if there’s something worth engaging about."""
        now = datetime.now()
        msgs = []
        
        
        # Example context signals (wire these to your sensors)
        # ctx = {
        #     "long_sitting": self.habits.too_long_since_break(),
        #     "approaching_bedtime": self.habits.is_bedtime_window(),
        #     "health_risk": False,
        # }

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