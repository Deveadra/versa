
import io
import sounddevice as sd
import numpy as np
import tempfile
import openai
from playsound import playsound
from elevenlabs import ElevenLabs

from assistant.config import settings

RATE = 16000  # 16kHz
CHANNELS = 1

def record_audio(duration: int = 5) -> str:
    """Record from microphone and save to wav temp file"""
    print("🎙️ Recording... speak now")
    audio = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS, dtype=np.int16)
    sd.wait()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    import soundfile as sf
    sf.write(temp.name, audio, RATE)
    return temp.name

def transcribe(file_path: str) -> str:
    """Send audio file to Whisper for STT"""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text.strip()

def speak(text: str):
    """TTS via ElevenLabs"""
    if not settings.eleven_api_key or not settings.eleven_voice_id:
        print("Ultron (text):", text)
        return
    client = ElevenLabs(api_key=settings.eleven_api_key)
    audio = client.generate(text=text, voice=settings.eleven_voice_id)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp.name, "wb") as f:
        f.write(audio)
    playsound(temp.name)
