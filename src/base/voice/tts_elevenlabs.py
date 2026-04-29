# src/base/voice/tts_elevenlabs.py
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from typing import Any, cast

try:
    import simpleaudio as sa
except ModuleNotFoundError:  # optional dependency for voice integrations
    sa = None  # type: ignore[assignment]
try:
    from elevenlabs.client import (
        ElevenLabs,  # <- matches the documented SDK usage :contentReference[oaicite:1]{index=1}
    )
except ModuleNotFoundError:  # optional dependency for voice integrations
    ElevenLabs = None  # type: ignore[assignment]

from config.config import settings

BytesLike = bytes | bytearray | memoryview
log = logging.getLogger(__name__)


def _require_simpleaudio() -> Any:
    if sa is None:
        raise RuntimeError("simpleaudio not installed")
    return cast(Any, sa)

# class Voice:
#     _instance: Voice | None = None
#     _current_playback = None
#     _lock = threading.Lock()

#     def __init__(self, model_id: str | None = None, output_format: str | None = None):
#         if not settings.eleven_api_key or not settings.eleven_voice_id:
#             raise RuntimeError("ElevenLabs API key/voice id not configured")

#         self.api_key = settings.eleven_api_key
#         self.voice_id = settings.eleven_voice_id

#         if not self.api_key:
#             raise RuntimeError("ElevenLabs API key not configured")

#         # Auto-resolve friendly names to IDs
#         if self.voice_id and not self.voice_id.isalnum():
#             try:
#                 client = ElevenLabs(api_key=self.api_key)
#                 voices = client.voices.list().voices
#                 match = next((v for v in voices if v.name == self.voice_id), None)
#                 if match:
#                     self.voice_id = match.voice_id
#                     logger.info(f"Resolved ElevenLabs voice '{match.name}' → {self.voice_id}")
#             except Exception as e:
#                 logger.warning(f"Failed to resolve voice name '{self.voice_id}': {e}")

#         self.client = ElevenLabs(api_key=settings.eleven_api_key)
#         self.voice_id = settings.eleven_voice_id
#         self.model_id = model_id or os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
#         self.output_format = output_format or os.getenv("ELEVENLABS_OUTPUT", "pcm_16000")


class Voice:
    _instance: Voice | None = None
    _current_playback: sa.PlayObject | None = None  # type: ignore[name-defined]
    _lock = threading.Lock()

    def __init__(self, model_id: str | None = None, output_format: str | None = None):
        _require_simpleaudio()

        if ElevenLabs is None:
            raise RuntimeError("ElevenLabs SDK not installed")

        if not settings.eleven_api_key or not settings.eleven_voice_id:
            raise RuntimeError("ElevenLabs API key/voice id not configured")

        self.api_key = settings.eleven_api_key
        self.client = ElevenLabs(api_key=self.api_key)

        self.model_id = model_id or os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.output_format = output_format or os.getenv("ELEVENLABS_OUTPUT", "pcm_16000")

        # resolve friendly name -> voice_id once, and keep it
        self.voice_id = self._resolve_voice_id(settings.eleven_voice_id)

    def _resolve_voice_id(self, voice: str) -> str:
        # If it's a friendly name (contains spaces / not an id), resolve via API
        if not voice:
            raise RuntimeError("ElevenLabs voice id not configured")
        if voice.isalnum():  # common voice_id format (e.g. JBFqnCBsd6RMkjVDRZzb)
            return voice

        try:
            resp = (
                self.client.voices.search()
            )  # documented pattern :contentReference[oaicite:2]{index=2}
            voices = getattr(resp, "voices", []) or []
            match = next((v for v in voices if getattr(v, "name", None) == voice), None)
            if match and getattr(match, "voice_id", None):
                log.info("Resolved ElevenLabs voice '%s' -> %s", voice, match.voice_id)
                return match.voice_id
        except Exception as e:
            log.warning("Failed to resolve voice name '%s': %s", voice, e)

        return voice  # fall back; if invalid, API call will fail later

    # Singleton access
    @classmethod
    def get_instance(cls) -> Voice:
        if cls._instance is None:
            cls._instance = Voice()
        return cls._instance

    def speak_async(self, text: str) -> None:
        """
        Synthesize and play audio asynchronously (non-blocking).
        """
        audio_bytes = self.synth(text)

        def _play():
            try:
                simpleaudio = _require_simpleaudio()
                with self._lock:
                    Voice._current_playback = simpleaudio.play_buffer(audio_bytes, 1, 2, 16000)
                playback = Voice._current_playback
                if playback is not None:
                    playback.wait_done()
            except Exception as e:
                print(f"[TTS Async] Failed to play audio: {e}")
            finally:
                with self._lock:
                    Voice._current_playback = None

        threading.Thread(target=_play, daemon=True).start()

    def stop_speaking(self) -> None:
        """
        Interrupt current speech playback (if any).
        """
        with self._lock:
            if Voice._current_playback:
                Voice._current_playback.stop()
                Voice._current_playback = None

    def speak_blocking(self, text: str) -> None:
        """
        Synthesize and play audio synchronously (blocking).
        """
        audio_bytes = self.synth(text)
        try:
            simpleaudio = _require_simpleaudio()
            play_obj = simpleaudio.play_buffer(audio_bytes, 1, 2, 16000)
            play_obj.wait_done()
        except Exception as e:
            print(f"[TTS Blocking] Failed to play audio: {e}")

    def speak(self, text: str) -> None:
        """
        Convenience wrapper: synthesize and play immediately.
        """
        audio_bytes = self.synth(text)
        try:
            simpleaudio = _require_simpleaudio()
            play_obj = simpleaudio.play_buffer(audio_bytes, 1, 2, 16000)
            play_obj.wait_done()
        except Exception as e:
            print(f"[TTS] Failed to play audio: {e}")

    def synth(self, text: str) -> bytes:
        result = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format=self.output_format,
        )
        return self._to_bytes(result)

    def synth_stream(self, text: str) -> Iterable[bytes]:
        result = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format=self.output_format,
        )
        if isinstance(result, (bytes, bytearray, memoryview)):
            yield bytes(result)
        else:
            for chunk in result:
                if chunk:
                    yield chunk

    @staticmethod
    def _to_bytes(data: BytesLike | Iterable[bytes]) -> bytes:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return b"".join(data)
