# src/base/voice/tts_ultron.py
from __future__ import annotations

import base64
import os
import tempfile

import requests
import simpleaudio as sa


class UltronVoice:
    _instance: UltronVoice | None = None

    def __init__(self, endpoint_url: str | None = None):
        self.endpoint_url = endpoint_url or os.getenv(
            "ULTRON_TTS_URL", "http://localhost:5000/speak"
        )

    @classmethod
    def get_instance(cls, endpoint_url: str | None = None) -> UltronVoice:
        if cls._instance is None:
            cls._instance = UltronVoice(endpoint_url)
        return cls._instance

    def speak(self, text: str) -> None:
        try:
            resp = requests.post(self.endpoint_url, json={"text": text}, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            audio_b64 = data.get("audio")
            if not audio_b64:
                raise ValueError("No audio field returned from TTS server.")

            audio_bytes = base64.b64decode(audio_b64)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                wave = sa.WaveObject.from_wave_file(tmp_path)
                play = wave.play()
                play.wait_done()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"[UltronVoice] Error during TTS: {e}")
