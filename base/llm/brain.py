
from __future__ import annotations
from typing import List, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from config.config import settings

from ..core.core import messages, reset_session, JARVIS_PROMPT, CURRENT_PERSONALITY, PERSONALITIES
from base.core.audio import stream_speak
from base.plugins import PLUGINS

_client = OpenAI(api_key=settings.openai_api_key)
_MODEL = settings.openai_model or "gpt-4o-mini"

class Brain:
    def __init__(self):
        self.client = _client
        self.model = _MODEL

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

def auto_set_personality(user_text: str) -> None:
    global CURRENT_PERSONALITY, JARVIS_PROMPT
    lowered = user_text.lower()
    if any(w in lowered for w in ["cpu", "memory", "system", "process"]):
        CURRENT_PERSONALITY = PERSONALITIES["sarcastic"]
    elif any(w in lowered for w in ["email", "calendar", "meeting", "schedule"]):
        CURRENT_PERSONALITY = PERSONALITIES["formal"]
    else:
        CURRENT_PERSONALITY = PERSONALITIES["default"]
    JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]

def ask_jarvis_stream(user_text: str) -> str:
    auto_set_personality(user_text)

    # quick plugin dispatch
    for name, fn in PLUGINS.items():
        if name in user_text.lower():
            return fn()

    messages.append({"role": "user", "content": user_text})

    # ✅ give Pylance the exact SDK type
    typed_msgs: List[ChatCompletionMessageParam] = cast(List[ChatCompletionMessageParam], messages)

    stream = _client.chat.completions.create(
        model=_MODEL,
        messages=typed_msgs,
        temperature=0.6,
        stream=True,
    )

    reply_accum = ""
    for chunk in stream:
        for choice in chunk.choices:
            token = choice.delta.content or ""
            if not token:
                continue
            reply_accum += token
            if token.endswith((".", "?", "!")):
                stream_speak(reply_accum.strip())
                reply_accum = ""

    if reply_accum:
        stream_speak(reply_accum.strip())
        messages.append({"role": "assistant", "content": reply_accum})

    return reply_accum
