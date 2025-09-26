
from __future__ import annotations
from typing import Iterable, Union, Optional
import os

from elevenlabs import ElevenLabs
from config.config import settings
import simpleaudio as sa
import threading


_voice = None
_currently_playing = None
_lock = threading.Lock()

BytesLike = Union[bytes, bytearray, memoryview]

class Voice:
    def __init__(self, model_id: Optional[str] = None, output_format: Optional[str] = None):
        if not settings.eleven_api_key or not settings.eleven_voice_id:
            raise RuntimeError("ElevenLabs API key/voice id not configured")

        self.client = ElevenLabs(api_key=settings.eleven_api_key)
        self.voice_id = settings.eleven_voice_id
        # sensible defaults; override via args or env if you want MP3/WAV instead
        self.model_id = model_id or os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.output_format = output_format or os.getenv("ELEVENLABS_OUTPUT", "pcm_16000")

    # Ultron can keep running tasks while speaking.
    def speak_async(text: str) -> None:
        """
        Synthesize and play audio asynchronously (non-blocking).
        Ultron can keep running while speaking.
        """
        global _voice, _current_playback
        if _voice is None:
            _voice = Voice()

        audio_bytes = _voice.synth(text)

        def _play():
            global _current_playback
            try:
                with _lock:
                    _current_playback = sa.play_buffer(audio_bytes, 1, 2, 16000)
                _current_playback.wait_done()  # runs in background thread
            except Exception as e:
                print(f"[TTS Async] Failed to play audio: {e}")
            finally:
                with _lock:
                    _current_playback = None

        threading.Thread(target=_play, daemon=True).start()

    def stop_speaking() -> None:
        """
        Interrupt current speech playback (if any).
        """
        global _current_playback
        with _lock:
            if _current_playback:
                _current_playback.stop()
                _current_playback = None
                
    # Ultron waits fully until playback finishes before continuing.
    def speak_blocking(text: str) -> None:
        """
        Synthesize and play audio synchronously.
        Ultron waits until playback finishes before continuing.
        """
        global _voice
        if _voice is None:
            _voice = Voice()

        audio_bytes = _voice.synth(text)
        try:
            play_obj = sa.play_buffer(audio_bytes, 1, 2, 16000)
            play_obj.wait_done()  # blocks until finished
        except Exception as e:
            print(f"[TTS Blocking] Failed to play audio: {e}")
            
    def speak(text: str) -> None:
        """
        Convenience wrapper: synthesize and play speech immediately.
        Uses ElevenLabs TTS under the hood.
        """
        global _voice
        if _voice is None:
            _voice = Voice()
        audio_bytes = _voice.synth(text)

        try:
            # play raw PCM 16kHz (since default output_format = pcm_16000)
            play_obj = sa.play_buffer(audio_bytes, 1, 2, 16000)
            play_obj.wait_done()
        except Exception as e:
            print(f"[TTS] Failed to play audio: {e}")
            
    def synth(self, text: str) -> bytes:
        """
        Synthesize to a single bytes object (always). If the SDK returns an iterator,
        we coalesce it into bytes.
        """
        result = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format=self.output_format,
        )
        return self._to_bytes(result)

    def synth_stream(self, text: str) -> Iterable[bytes]:
        """
        Stream synthesis chunks. If the SDK returns bytes (non-streaming),
        yield it once; otherwise yield chunks as provided.
        """
        result = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format=self.output_format,
        )
        if isinstance(result, (bytes, bytearray, memoryview)):
            yield bytes(result)
        else:
            # assume it's an iterable/iterator of bytes
            for chunk in result:
                if chunk:
                    yield chunk

    @staticmethod
    def _to_bytes(data: Union[BytesLike, Iterable[bytes]]) -> bytes:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return b"".join(data)
