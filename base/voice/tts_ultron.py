
import requests
import os
import tempfile
import pygame  # for simple playback, since you already use audio output

class UltronVoice:
    _instance = None

    def __init__(self, endpoint_url: str = None):
        # Default to env var, fallback to localhost
        self.endpoint_url = endpoint_url or os.getenv("ULTRON_TTS_URL", "http://localhost:5000/speak")

        # init pygame mixer once
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    @classmethod
    def get_instance(cls, endpoint_url: str = None):
        if cls._instance is None:
            cls._instance = UltronVoice(endpoint_url)
        return cls._instance

    def speak(self, text: str) -> None:
        """Send text to Ultron TTS API and play the resulting audio."""
        try:
            resp = requests.post(self.endpoint_url, json={"text": text}, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "audio" not in data:
                raise ValueError("No audio field returned from TTS server.")

            # decode base64 audio into temp file
            import base64
            audio_bytes = base64.b64decode(data["audio"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # play via pygame
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.remove(tmp_path)

        except Exception as e:
            print(f"[UltronVoice] Error during TTS: {e}")
