# base/llm/brain.py
from __future__ import annotations

import os
from typing import cast

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

try:
    from base.core.audio import stream_speak
except Exception:

    def stream_speak(*args, **kwargs):
        return None


from base.voice.tts_elevenlabs import Voice
from config.config import settings

from ..core.core import PERSONALITIES, messages

# ---------- OpenAI client / model ----------
_CLIENT = OpenAI(api_key=settings.openai_api_key)
_MODEL = settings.openai_model or os.getenv("BRAIN_MODEL", "gpt-4o-mini")


def _check_vocal_cue(user_text: str) -> str | None:
    lowered = user_text.lower()
    wake_word = getattr(settings, "wake_word", "").lower()
    wake_commands = getattr(settings, "wake_commands", {})

    if wake_word and lowered.startswith(wake_word):
        for phrase, command in wake_commands.items():
            if phrase in lowered:
                return command
    return None


class Brain:
    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        self.client = client or _CLIENT
        self.model = model or _MODEL
        try:
            self.voice = Voice.get_instance()
        except Exception as exc:
            logger.warning("Voice disabled: optional TTS dependencies unavailable (%s)", exc)
            self.voice = None

    # ---------- helpers ----------
    def _set_system_prompt(self, prompt: str) -> None:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = prompt
        else:
            messages.insert(0, {"role": "system", "content": prompt})

    def _speak_async(self, text: str) -> None:
        if not text or not getattr(settings, "auto_speak", False):
            return

        try:
            speak_async = getattr(self.voice, "speak_async", None)
            if callable(speak_async):
                speak_async(text)
            else:
                stream_speak(text)
        except Exception:
            logger.exception("auto_speak failed")

    # ---------- persona ----------
    def auto_set_personality(self, user_text: str) -> None:
        lowered = user_text.lower()

        if any(word in lowered for word in ("cpu", "memory", "system", "process")):
            personality = PERSONALITIES["sarcastic"]
        elif any(word in lowered for word in ("email", "calendar", "meeting", "schedule")):
            personality = PERSONALITIES["formal"]
        else:
            personality = PERSONALITIES["default"]

        self._set_system_prompt(personality["prompt"])

    # ---------- ask (text or json) ----------
    def ask_brain(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_format: str = "text",
    ) -> str:
        """
        Send a prompt to OpenAI. Supports plain text replies and JSON-only replies.
        """
        try:
            # Voice toggle commands should happen before the API call.
            command = _check_vocal_cue(prompt)
            if command == "disable_speak":
                settings.auto_speak = False
                return "Understood. I’ll stop speaking and switch to text."
            if command == "enable_speak":
                settings.auto_speak = True
                return "Voice enabled again."

            msg_list: list[dict[str, str]] = []

            if system_prompt:
                msg_list.append({"role": "system", "content": system_prompt})
            elif messages and messages[0].get("role") == "system":
                msg_list.append(
                    {
                        "role": "system",
                        "content": str(messages[0].get("content", "")),
                    }
                )

            if response_format == "json":
                msg_list.append({"role": "system", "content": "Respond ONLY in strict JSON."})

            msg_list.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], msg_list),
                temperature=0.6,
            )

            reply = (completion.choices[0].message.content or "").strip()

            if reply:
                self._speak_async(reply)

            return reply or "Okay."

        except Exception as e:
            logger.exception(f"[ask_brain error] {e}")
            return "Sorry, I couldn’t process that."

    # ---------- streaming ask with TTS ----------
    def ask_aerith_stream(self, user_text: str) -> str:
        try:
            self.auto_set_personality(user_text)

            # Voice toggle commands should happen before the API call.
            command = _check_vocal_cue(user_text)
            if command == "disable_speak":
                settings.auto_speak = False
                return "Understood. I’ll stop speaking and switch to text."
            if command == "enable_speak":
                settings.auto_speak = True
                return "Voice enabled again."

            # Ensure a system prompt exists and gather recent conversation history.
            system_message = (
                messages[0]
                if messages and messages[0].get("role") == "system"
                else {"role": "system", "content": PERSONALITIES["default"]["prompt"]}
            )

            recent_history = [
                m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")
            ][-10:]

            payload = [system_message, *recent_history, {"role": "user", "content": user_text}]
            typed_msgs = cast(list[ChatCompletionMessageParam], payload)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=typed_msgs,
                temperature=0.6,
                stream=True,
            )

            full_reply = ""
            speech_buffer = ""

            for chunk in stream:
                for choice in chunk.choices:
                    token = choice.delta.content or ""
                    if not token:
                        continue

                    full_reply += token
                    speech_buffer += token

                    if getattr(settings, "auto_speak", False) and token.endswith((".", "?", "!")):
                        stream_speak(speech_buffer.strip())
                        speech_buffer = ""

            if speech_buffer and getattr(settings, "auto_speak", False):
                stream_speak(speech_buffer.strip())

            full_reply = full_reply.strip()

            messages.append({"role": "user", "content": user_text})
            if full_reply:
                messages.append({"role": "assistant", "content": full_reply})

            return full_reply or "Okay."

        except Exception as e:
            logger.error(f"[stream error] {e}")
            return "Something went wrong during streaming."


# ---------- Singleton + module-level wrappers ----------
_brain = Brain()


def ask_brain(
    prompt: str,
    system_prompt: str | None = None,
    response_format: str = "text",
) -> str:
    return _brain.ask_brain(
        prompt,
        system_prompt=system_prompt,
        response_format=response_format,
    )


def ask_aerith_stream(user_text: str) -> str:
    return _brain.ask_aerith_stream(user_text)
