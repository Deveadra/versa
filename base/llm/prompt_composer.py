from __future__ import annotations
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

def compose_prompt(system_prompt: str,
                    user_text: str,
                    persona_text: Optional[str] = None,
                    memories: Optional[List[Dict[str, Any]]] = None,
                    extra_context: Optional[str] = None,
                    top_k_memories: int = 3) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(f"SYSTEM:\\n{system_prompt.strip()}\\n")
    persona_block = compose_persona_block(persona_text)
    if persona_block:
        parts.append(persona_block)
    if memories:
        parts.append(compose_retrieval_block(memories, top_k=top_k_memories))
    if extra_context:
        parts.append("Context:\\n" + extra_context.strip() + "\\n")
    parts.append("User:\\n" + (user_text or "").strip() + "\\n")
    return "\\n".join(parts)
