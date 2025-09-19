
from __future__ import annotations
import subprocess, openai, time
import os

from ..core.core import messages, reset_session, JARVIS_PROMPT, CURRENT_PERSONALITY, PERSONALITIES
from base.core.audio import stream_speak
from base.plugins import PLUGINS
from openai import OpenAI
from assistant.config.config import settings

openai.api_key = os.getenv("OPENAI_API_KEY")

class Brain:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    
def auto_set_personality(user_text):
    global CURRENT_PERSONALITY, JARVIS_PROMPT
    lowered = user_text.lower()
    if any(word in lowered for word in ["cpu", "memory", "system", "process"]):
        CURRENT_PERSONALITY = PERSONALITIES["sarcastic"]
    elif any(word in lowered for word in ["email", "calendar", "meeting", "schedule"]):
        CURRENT_PERSONALITY = PERSONALITIES["formal"]
    else:
        CURRENT_PERSONALITY = PERSONALITIES["default"]
    JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]


def ask_jarvis_stream(user_text):
    auto_set_personality(user_text)
    for plugin in PLUGINS:
        if plugin in user_text.lower():
            return PLUGINS[plugin]()
    messages.append({"role": "user", "content": user_text})
    response_stream = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True)
    reply_accum = ""
    for chunk in response_stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                reply_accum += token
                if any(p in token for p in [".", "?", "!"]):
                    stream_speak(reply_accum.strip())
                    reply_accum = ""
    if reply_accum:
        messages.append({"role": "assistant", "content": reply_accum})
    return reply_accum