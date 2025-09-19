
from __future__ import annotations
from elevenlabs import ElevenLabs
from assistant.config.config import settings

class Voice:
    def __init__(self):
        if not settings.eleven_api_key or not settings.eleven_voice_id:
            raise RuntimeError("ElevenLabs API key/voice id not configured")
        self.client = ElevenLabs(api_key=settings.eleven_api_key)
        self.voice_id = settings.eleven_voice_id

    def synth(self, text: str) -> bytes:
        audio = self.client.text_to_speech.convert(voice_id=self.voice_id, text=text)
        return audio  # bytes; you can write to .wav or stream