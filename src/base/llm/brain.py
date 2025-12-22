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
    # def ask_jarvis_stream(self, user_text: str) -> str:
    #     import time
    #     from loguru import logger

    #     self.auto_set_personality(user_text)

    #     # Quick plugin dispatch
    #     for name, fn in PLUGINS.items():
    #         if name in user_text.lower():
    #             return fn()

    #     # Voice toggle commands
    #     command = _check_vocal_cue(user_text)
    #     if command == "disable_speak":
    #         settings.auto_speak = False
    #         return "Understood. I’ll stop speaking and switch to text."
    #     elif command == "enable_speak":
    #         settings.auto_speak = True
    #         return "Voice enabled again."

    #     # Validate and sanitize conversation history
    #     safe_messages = []
    #     for m in messages:
    #         if not m.get("content"):
    #             logger.warning(f"Removed empty message from history: {m}")
    #             continue
    #         safe_messages.append(m)

    #     if not safe_messages or safe_messages[0]["role"] != "system":
    #         system_prompt = messages[0].get("content", PERSONALITIES["default"]["prompt"])
    #         safe_messages.insert(0, {"role": "system", "content": system_prompt})

    #     # Context trimming
    #     MAX_PAIRS = 10
    #     placeholder = {"role": "system", "content": "[Previous conversation truncated]"}
    #     placeholder_present = len(safe_messages) > 1 and "truncated" in safe_messages[1]["content"]

    #     while len(safe_messages) - 1 > 2 * MAX_PAIRS:
    #         if placeholder_present:
    #             safe_messages.pop(2); safe_messages.pop(2)
    #         else:
    #             safe_messages.pop(1); safe_messages.pop(1)
    #             safe_messages.insert(1, placeholder)
    #             placeholder_present = True

    #     # API call with retry logic
    #     user_msg = {"role": "user", "content": user_text}
    #     payload = safe_messages + [user_msg]

    #     try:
    #         stream = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=payload,
    #             temperature=0.6,
    #             stream=True,
    #         )
    #     except Exception as e:
    #         logger.error(f"LLM error: {e}, retrying once...")
    #         time.sleep(2)
    #         try:
    #             stream = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=payload,
    #                 temperature=0.6,
    #                 stream=True,
    #             )
    #         except Exception as e2:
    #             logger.error(f"Retry failed: {e2}")
    #             if settings.auto_speak:
    #                 stream_speak("I'm sorry, I couldn’t process that.")
    #             return None

    #     # Streaming and buffering
    #     full_reply = ""
    #     buffer = ""
    #     error_happened = False

    #     try:
    #         for chunk in stream:
    #             for choice in chunk.choices:
    #                 token = choice.delta.content or ""
    #                 if not token:
    #                     continue
    #                 full_reply += token
    #                 buffer += token
    #                 if token.endswith(('.', '?', '!')):
    #                     stream_speak(buffer.strip())
    #                     buffer = ""
    #     except Exception as err:
    #         error_happened = True
    #         logger.error(f"Stream interrupted: {err}")

    #     if buffer:
    #         stream_speak(buffer.strip())

    #     full_reply = full_reply.strip()

    #     if error_happened:
    #         if full_reply and settings.auto_speak:
    #             stream_speak("Sorry, I lost connection.")
    #         elif not full_reply and settings.auto_speak:
    #             stream_speak("I'm sorry, I couldn't process that.")
    #         return None

    #     # Update global history
    #     messages.clear()
    #     messages.extend(safe_messages)
    #     messages.append(user_msg)
    #     messages.append({"role": "assistant", "content": full_reply})

    #     return full_reply
    
    
    
    def ask_jarvis_stream(self, user_text: str) -> str:
        import time
        from loguru import logger

        self.auto_set_personality(user_text)

<<<<<<< Updated upstream
        for name, fn in PLUGINS.items():
            if name in user_text.lower():
                return fn() or ""

        command = _check_vocal_cue(user_text)
        if command == "disable_speak":
            settings.auto_speak = False
            return "Understood. I’ll stop speaking and switch to text."
        elif command == "enable_speak":
            settings.auto_speak = True
            return "Voice enabled again."

        safe_messages = []
        for m in messages:
            if not m.get("content"):
                logger.warning(f"Removed empty message from history: {m}")
                continue
            safe_messages.append(m)

        if not safe_messages or safe_messages[0]["role"] != "system":
            system_prompt = messages[0].get("content", PERSONALITIES["default"]["prompt"])
            safe_messages.insert(0, {"role": "system", "content": system_prompt})

        MAX_PAIRS = 10
        placeholder = {"role": "system", "content": "[Previous conversation truncated]"}
        placeholder_present = len(safe_messages) > 1 and "truncated" in safe_messages[1]["content"]

        while len(safe_messages) - 1 > 2 * MAX_PAIRS:
            if placeholder_present:
                safe_messages.pop(2)
                safe_messages.pop(2)
            else:
                safe_messages.pop(1)
                safe_messages.pop(1)
                safe_messages.insert(1, placeholder)
                placeholder_present = True

        user_msg = {"role": "user", "content": user_text}
        payload = safe_messages + [user_msg]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=0.6,
                stream=True,
            )
        except Exception as e:
            logger.error(f"LLM error: {e}, retrying once...")
            time.sleep(2)
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=0.6,
                    stream=True,
                )
            except Exception as e2:
                logger.error(f"Retry failed: {e2}")
                if settings.auto_speak:
                    stream_speak("I'm sorry, I couldn’t process that.")
                return "I'm sorry, I couldn’t process that."

        full_reply = ""
        buffer = ""
        error_happened = False

        try:
            for chunk in stream:
                for choice in chunk.choices:
                    token = choice.delta.content or ""
                    if not token:
                        continue
                    full_reply += token
                    buffer += token
                    if token.endswith(('.', '?', '!')):
                        stream_speak(buffer.strip())
                        buffer = ""
        except Exception as err:
            error_happened = True
            logger.error(f"Stream interrupted: {err}")

        if buffer:
            stream_speak(buffer.strip())

        full_reply = full_reply.strip()

        if error_happened:
            if full_reply and settings.auto_speak:
                stream_speak("Sorry, I lost connection.")
            elif not full_reply and settings.auto_speak:
                stream_speak("I'm sorry, I couldn't process that.")
            return full_reply or "Sorry, I lost connection."

        messages.clear()
        messages.extend(safe_messages)
        messages.append(user_msg)
        messages.append({"role": "assistant", "content": full_reply})

        return full_reply or ""

    
    
=======
        # Add user message
        messages.append({"role": "user", "content": user_text})

        # Filter last N user/assistant turns, and enforce system prompt
        safe_messages = [m for m in messages if m.get("role") in ("user", "assistant")][-10:]

        if not safe_messages or safe_messages[0]["role"] != "system":
            system_prompt = PERSONALITIES["default"]["prompt"]
            safe_messages.insert(0, {"role": "system", "content": system_prompt})

        typed_msgs: list[ChatCompletionMessageParam] = cast(
            list[ChatCompletionMessageParam], safe_messages
        )

        try:
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

            return reply_accum.strip() or "Okay."
        except Exception as e:
            logger.error(f"[stream error] {e}")
            return "Something went wrong during streaming."
>>>>>>> Stashed changes

    # def ask_jarvis_stream(self, user_text: str) -> str:
    #     self.auto_set_personality(user_text)

    #     # quick plugin dispatch
    #     for name, fn in PLUGINS.items():
    #         if name in user_text.lower():
    #             return fn()

    #     messages.append({"role": "user", "content": user_text})
    #     typed_msgs: list[ChatCompletionMessageParam] = cast(
    #         list[ChatCompletionMessageParam], messages
    #     )

    #     stream = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=typed_msgs,
    #         temperature=0.6,
    #         stream=True,
    #     )

    #     reply_accum = ""
    #     for chunk in stream:
    #         for choice in chunk.choices:
    #             token = choice.delta.content or ""
    #             if not token:
    #                 continue
    #             reply_accum += token
    #             if token.endswith((".", "?", "!")):
    #                 stream_speak(reply_accum.strip())
    #                 reply_accum = ""

    #     if reply_accum:
    #         stream_speak(reply_accum.strip())
    #         messages.append({"role": "assistant", "content": reply_accum})

    #     return reply_accum


# ---------- Singleton + module-level wrappers ----------
_brain = Brain()


def ask_brain(prompt: str, system_prompt: str | None = None, response_format: str = "text") -> str:
    return _brain.ask_brain(prompt, system_prompt=system_prompt, response_format=response_format)


def ask_jarvis_stream(user_text: str) -> str:
    return _brain.ask_jarvis_stream(user_text)
