SYSTEM_PROMPT = (
    "You are Ultron, a helpful, personal assistant. Use supplied memories as context. "
    "If information is missing, ask concise follow-ups. Be fast, precise, and personable."
)

# def build_prompt(memories: list[str], user: str) -> str:
#     ctx = "\n".join(f"- {m}" for m in memories)
#     prefix = f"Here are relevant memories about the user (may be partial):\n{ctx}\n\n" if memories else ""
#     return prefix + f"User: {user}\nAssistant:"


def build_prompt(memories, user_text, extra_context=""):
    context_block = "\n".join(memories)
    return f"""User said: {user_text}


Relevant memories:
{context_block}


{extra_context}
"""
