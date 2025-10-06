# base/llm/brain.py
from __future__ import annotations

import os
from typing import cast

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from base.core.audio import stream_speak
from base.plugins import PLUGINS
from base.voice.tts_elevenlabs import Voice
from config.config import settings

from ..core.core import PERSONALITIES, messages

# ---------- OpenAI client / model ----------
_CLIENT = OpenAI(api_key=settings.openai_api_key)  # new SDK
_MODEL = settings.openai_model or os.getenv("BRAIN_MODEL", "gpt-4o-mini")
# complete = cast(OpenAI, _CLIENT).chat.completions.create


def _check_vocal_cue(user_text: str) -> str | None:
    lowered = user_text.lower()
    if lowered.startswith(settings.wake_word.lower()):
        for phrase, command in settings.wake_commands.items():
            if phrase in lowered:
                return command
    return None


class Brain:
    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        self.client = client or _CLIENT
        self.model = model or _MODEL
        self.voice = Voice.get_instance()

    # -------- persona -----------
    def auto_set_personality(self, user_text: str) -> None:
        global CURRENT_PERSONALITY, JARVIS_PROMPT
        lowered = user_text.lower()
        if any(w in lowered for w in ["cpu", "memory", "system", "process"]):
            CURRENT_PERSONALITY = PERSONALITIES["sarcastic"]
        elif any(w in lowered for w in ["email", "calendar", "meeting", "schedule"]):
            CURRENT_PERSONALITY = PERSONALITIES["formal"]
        else:
            CURRENT_PERSONALITY = PERSONALITIES["default"]
        JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]

    # -------- ask (text or json) -----------
    def ask_brain(
        self, prompt: str, system_prompt: str | None = None, response_format: str = "text"
    ) -> str:
        """
        Send a prompt to OpenAI. Supports text or JSON output.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if response_format == "json":
                messages.append({"role": "system", "content": "Respond ONLY in strict JSON."})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
            )

            reply = (completion.choices[0].message.content or "").strip()

            # vocal cue handling
            command = _check_vocal_cue(prompt)
            if command == "disable_speak":
                settings.auto_speak = False
                return "Understood. I’ll stop speaking and switch to text."
            elif command == "enable_speak":
                settings.auto_speak = True
                return "Voice enabled again."

            # auto-speak if enabled
            if settings.auto_speak:
                self.voice.speak_async(reply)

            # brain.py
            if getattr(settings, "auto_speak", False) and hasattr(self, "voice"):
                speak_async = getattr(self.voice, "speak_async", None)
                if callable(speak_async):
                    speak_async(reply)


            return reply
        except Exception as e:
            logger.exception(f"[ask_brain error] {e}")
            return "Sorry, I couldn’t process that."

    # -------- streaming ask with TTS -----------
    def ask_jarvis_stream(self, user_text: str) -> str:
        self.auto_set_personality(user_text)

        # quick plugin dispatch
        for name, fn in PLUGINS.items():
            if name in user_text.lower():
                return fn()

        messages.append({"role": "user", "content": user_text})
        typed_msgs: list[ChatCompletionMessageParam] = cast(
            list[ChatCompletionMessageParam], messages
        )

        stream = self.client.chat.completions.create(
            model=self.model,
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


# ---------- Singleton + module-level wrappers ----------
_brain = Brain()


def ask_brain(prompt: str, system_prompt: str | None = None, response_format: str = "text") -> str:
    return _brain.ask_brain(prompt, system_prompt=system_prompt, response_format=response_format)


def ask_jarvis_stream(user_text: str) -> str:
    return _brain.ask_jarvis_stream(user_text)
