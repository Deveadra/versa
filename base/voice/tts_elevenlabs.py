
from __future__ import annotations
from typing import Iterable, Union, Optional
import os

from elevenlabs import ElevenLabs
from config.config import settings

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
