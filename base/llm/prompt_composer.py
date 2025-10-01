from __future__ import annotations

from textwrap import dedent
from typing import List, Dict, Any, Optional

import datetime
import os

from datetime import datetime
from base.core.profile_manager import ProfileManager
from base.memory.store import MemoryStore
from base.learning.habit_miner import HabitMiner
from base.learning.sentiment import quick_polarity, quick_polarity_label
from base.personality.tone_adapter import ToneAdapter



def _format_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return "unknown time"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
    except Exception:
        return ts.split("T")[0] if "T" in ts else ts


def justify_memory(mem: Dict[str, Any]) -> str:
    if not mem:
        return ""
    parts = []
    if mem.get("source"):
        parts.append(f"from {mem['source']}")
    if mem.get("score") is not None:
        try:
            parts.append(f"score={float(mem['score']):.2f}")
        except Exception:
            parts.append(f"score={mem.get('score')}")
    ts = mem.get("last_used") or mem.get("created_at") or mem.get("timestamp")
    if ts:
        parts.append(f"as of {_format_timestamp(ts)}")
    return ", ".join(parts) if parts else "relevant memory"

def compose_persona_block(persona_text: Optional[str]) -> str:
    if not persona_text:
        return ""
    lines = [l.strip() for l in persona_text.splitlines() if l.strip()][:6]
    if not lines:
        return ""
    return "Persona:\\n" + "\\n".join(lines) + "\\n"

def compose_habit_block(habits: list[str], top_k: int = 3) -> str:
    """
    Turn a list of habit summaries into a compact prompt block.
    Example habit: "User usually plays lo-fi music at night"
    """
    if not habits:
        return ""
    # keep top habits (e.g. most frequent / most recent)
    selected = habits[:top_k]
    lines = ["Observed habits:"]
    for i, h in enumerate(selected, start=1):
        lines.append(f"{i}. {h}")
    return "\n".join(lines) + "\n"

def compose_retrieval_block(memories: List[Dict[str, Any]], top_k: int = 3) -> str:
    if not memories:
        return ""
    def _key(m: Dict[str, Any]):
        score = m.get("score") or 0.0
        ts = m.get("last_used") or m.get("created_at") or m.get("timestamp") or ""
        return (score, ts)
    sorted_mem = sorted(memories, key=_key, reverse=True)
    selected = sorted_mem[:top_k]
    lines = ["Relevant memories:"]
    for i, m in enumerate(selected, start=1):
        summary = m.get("summary") or m.get("text") or "<no summary available>"
        justification = justify_memory(m)
        lines.append(f"{i}. {summary}  ({justification})")
    return "\\n".join(lines) + "\\n"


def synthesize_memories(memories: List[Dict[str, Any]], top_k: int = 3) -> str:
    """
    Turn raw memory dicts into a natural narrative Ultron can use.
    """
    if not memories:
        return ""
    def _key(m: Dict[str, Any]):
        score = m.get("score") or 0.0
        ts = m.get("last_used") or m.get("created_at") or m.get("timestamp") or ""
        return (score, ts)
    selected = sorted(memories, key=_key, reverse=True)[:top_k]

    lines = []
    for m in selected:
        summary = m.get("summary") or m.get("text") or "<no summary>"
        lines.append(f"- {summary} ({justify_memory(m)})")

    return "You recall the following about the user:\n" + "\n".join(lines) + "\n"


def compose_prompt(
    system_prompt: str,
    user_text: str,
    profile_mgr: ProfileManager,
    memory_store: MemoryStore,
    habit_miner: HabitMiner,
    persona_text: str | None = None,
    memories: list[dict[str, Any]] | None = None,
    habits: list[str] | None = None,
    extra_context: str | None = None,
    top_k_memories: int = 3,
    top_k_habits: int = 3,
    channel: str = "text",         # NEW
    include_kg: bool = True,       # optional future hook
) -> str:
    """
    Compose a dynamic prompt that fuses:
    - Ultron's evolving persona
    - Habits and preferences
    - Relevant memories
    - Sentiment/tone analysis
    """
    parts: list[str] = []

    # Core system rules
    if system_prompt:
        parts.append(f"SYSTEM:\n{system_prompt.strip()}\n")

    # Persona (dynamic, evolving)
    if persona_text:
        parts.append(f"You are Ultron. {persona_text.strip()}\n")

    persona_block = compose_persona_block(persona_text)
    if persona_block:
        parts.append(persona_block)

    # # Habits / preferences (auto-mined)
    # habits = habit_miner.export_summary()
    if habits:
        parts.append(compose_habit_block(habits, top_k=top_k_habits))

    if memories:
        parts.append(compose_retrieval_block(memories, top_k=top_k_memories))

    if extra_context:
        parts.append("Context:\n" + extra_context.strip() + "\n")

    # Memory synthesis (recent & relevant)
    mems = memory_store.keyword_search(user_text, limit=top_k_memories)
    if mems:
        mem_dicts = [{"text": m} for m in mems]
        parts.append(synthesize_memories(mem_dicts, top_k=top_k_memories))

    # NEW: style plan (sentiment + bandit + channel + habits)
    style_plan = compose_style_plan(user_text, profile_mgr, habit_miner, channel=channel)
    parts.append(render_style_instructions(style_plan))

    # Sentiment analysis
    polarity = quick_polarity(user_text)   # keep float for numeric analysis
    if polarity:
        tone_hint = ToneAdapter.adapt(quick_polarity_label(user_text))  # pass str instead
        parts.append(f"Adjust your tone: {tone_hint}\n")

    # Extra session context
    if extra_context:
        parts.append("Context:\n" + extra_context.strip() + "\n")

    # User input
    parts.append("User:\n" + (user_text or '').strip() + "\n")

    return "\n".join(parts)


def compose_style_plan(
    user_text: str,
    profile_mgr: ProfileManager,
    habit_miner: HabitMiner,
    channel: str = "text",  # "voice" or "text"
) -> Dict[str, Any]:
    """
    Produce a concrete style plan Ultron follows for *this* response.
    Combines: sentiment, policy bandit, habits, and channel.
    """
    profile = profile_mgr.load_profile()  # assume it returns dict
    polarity = quick_polarity(user_text)

    # basic time-of-day vibe
    hour = datetime.now().hour
    tod = "morning" if 5 <= hour < 12 else "afternoon" if hour < 18 else "evening"

    # defaults
    polarity_label = quick_polarity_label(user_text)  
    bandit = ToneAdapter(profile)
    policy = bandit.choose_policy()

    plan = {
        "tone_hint": ToneAdapter.adapt(polarity_label),
        "style_id": policy["id"],
        "max_words": policy["max_words"],
        "formality": "casual" if policy["id"] in ("casual", "playful") else "neutral",
        "humor": policy["id"] == "playful",
        "brevity": policy["id"] == "succinct",
        "channel": channel,
        "time_of_day": tod,
        "tts": {"wpm": 170, "pause_ms": 120, "filler_ok": False},
    }

    # adapt for sentiment
    if polarity == "negative":
        plan["humor"] = False
        plan["formality"] = "warm"
        plan["tts"]["wpm"] = 150
        plan["tts"]["pause_ms"] = 160
    elif polarity == "positive":
        plan["tts"]["wpm"] = 185
        plan["tts"]["pause_ms"] = 100

    # adapt for channel (voice vs text)
    if channel == "voice":
        # very light disfluency is more natural in voice; keep it optional
        plan["tts"]["filler_ok"] = True
        # cap length harder in voice mode
        plan["max_words"] = min(plan["max_words"], 100)

    # adapt for habits/preferences if present
    habits = habit_miner.export_summary()
    if "Dislikes long answers" in (habits or ""):
        plan["brevity"] = True
        plan["max_words"] = min(plan["max_words"], 90)

    return plan


def render_style_instructions(plan: Dict[str, Any]) -> str:
    """
    Turn the style plan into explicit, LLM-friendly instructions.
    """
    bullets = [
        f"Tone guide: {plan['tone_hint']}",
        f"Style: {plan['style_id']} (formality={plan['formality']}, humor={'on' if plan['humor'] else 'off'}, brevity={'on' if plan['brevity'] else 'off'})",
        f"Max words: {plan['max_words']}",
        f"Channel: {plan['channel']} (time_of_day={plan['time_of_day']})",
    ]
    if plan["channel"] == "voice":
        tts = plan["tts"]
        bullets.append(f"TTS: ~{tts['wpm']} wpm, pause≈{tts['pause_ms']}ms, filler_ok={'yes' if tts['filler_ok'] else 'no'}")
        bullets.append("If voice: keep sentences shorter, use natural pauses instead of lists when possible.")

    return "Style plan:\n- " + "\n- ".join(bullets) + "\n"