from __future__ import annotations

from textwrap import dedent
from typing import List, Dict, Any, Optional

import datetime


def _format_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return "unknown time"
    try:
        dt = datetime.datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ts.split("T")[0] if "T" in ts else ts.split(" ")[0]

def justify_memory(mem: Dict[str, Any]) -> str:
    if not mem:
        return ""
    if mem.get("reason"):
        return mem["reason"]
    parts = []
    src = mem.get("source")
    if src:
        parts.append(f"from {src}")
    if mem.get("score") is not None:
        try:
            parts.append(f"score={float(mem['score']):.2f}")
        except Exception:
            parts.append(f"score={mem.get('score')}")
    ts = mem.get("last_used") or mem.get("created_at") or mem.get("timestamp")
    if ts:
        parts.append(f"recent={_format_timestamp(ts)}")
    if parts:
        return ", ".join(parts)
    return "relevant memory"

# def compose_persona_block(profile_mgr: ProfileManager) -> str:
#     """
#     Build the personality + tone block.
#     This evolves as Ultron learns from HabitMiner and profile enrichment.
#     """
#     persona = profile_mgr.get_current_persona()
#     lines = [
#         f"Ultron Personality: {persona.get('name', 'Ultron')}",
#         f"Style: {persona.get('style', 'formal but warm')}",
#         f"Tone Weights: {persona.get('tone_weights', {})}",
#         f"Known User Prefs: {persona.get('user_prefs', {})}"
#     ]
#     return "\n".join(lines)

def compose_persona_block(persona_text: Optional[str]) -> str:
    if not persona_text:
        return ""
    lines = [l.strip() for l in persona_text.splitlines() if l.strip()][:6]
    if not lines:
        return ""
    return "Persona:\\n" + "\\n".join(lines) + "\\n"

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

# def compose_retrieval_block(memories: List[str], facts: List[str]) -> str:
#     """
#     Include relevant memories and facts for grounding the model’s reply.
#     """
#     block = ["--- Contextual Memory ---"]
#     if facts:
#         block.append("Facts:")
#         for f in facts:
#             block.append(f" - {f}")
#     if memories:
#         block.append("Recent Memories:")
#         for m in memories:
#             block.append(f" - {m}")
#     return "\n".join(block)

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
        ts = m.get("last_used") or m.get("created_at") or m.get("timestamp")
        when = _format_timestamp(ts)
        lines.append(f"- {summary} (as of {when})")

    return "You recall the following about the user:\n" + "\n".join(lines) + "\n"


def compose_prompt(
    system_prompt: str,
    user_text: str,
    persona_text: Optional[str] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
    extra_context: Optional[str] = None,
    tone_hint: Optional[str] = None,
    top_k_memories: int = 3,
) -> str:
    """
    Compose the full LLM prompt with persona, memory, and user context.
    """
    parts: List[str] = []

    if system_prompt:
        parts.append(f"SYSTEM:\n{system_prompt.strip()}\n")

    # Persona (dynamic, evolving)
    if persona_text:
        parts.append(f"You are Ultron. {persona_text.strip()}\n")

    # Learned habits / memories
    if memories:
        parts.append(synthesize_memories(memories, top_k=top_k_memories))

    # Extra context (session-level cues, e.g. "user seems tired")
    if extra_context:
        parts.append("Context:\n" + extra_context.strip() + "\n")

    # Tone hint (from sentiment analysis or personality adapters)
    if tone_hint:
        parts.append(f"Adjust your tone: {tone_hint}\n")

    # User input (always last)
    parts.append("User:\n" + (user_text or "").strip() + "\n")

    return "\n".join(parts)


# def compose_prompt(
#     user_text: str,
#     profile_mgr: ProfileManager,
#     memory_store: MemoryStore,
#     retrieved: Optional[Dict[str, List[str]]] = None,
# ) -> str:
#     """
#     Assemble the full LLM prompt:
#     - Persona
#     - Contextual retrieval (facts + memories)
#     - User input
#     """
#     persona_block = compose_persona_block(profile_mgr)

#     facts = [f"{k}: {v}" for (k, v) in memory_store.list_facts()]
#     memories = []
#     if retrieved and "events" in retrieved:
#         memories = retrieved["events"]

#     retrieval_block = compose_retrieval_block(memories, facts)

#     return dedent(f"""
#     {persona_block}

#     {retrieval_block}

#     --- Conversation ---
#     User: {user_text}
#     Ultron:
#     """).strip()