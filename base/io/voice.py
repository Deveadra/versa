
import os
import io
import sounddevice as sd
import numpy as np
import tempfile
import openai
from playsound import playsound
from elevenlabs import ElevenLabs

from config.config import settings

# Tune these once, globally
SD_RATE = 16000      # 16 kHz is fine for voice and low-latency
SD_CHANNELS = 1
SD_DTYPE = "int16"   # match the server's PCM format


# sd.default.latency = ("low", "low")
# For PCM 16k mono little-endian:
# pcm = _to_bytes(result)
# sa.play_buffer(pcm, num_channels=1, bytes_per_sample=2, sample_rate=16000)  # non-blocking



def record_audio(duration: int = 5) -> str:
    """Record from microphone and save to wav temp file"""
    print("🎙️ Recording... speak now")
    audio = sd.rec(int(duration * SD_RATE), samplerate=SD_RATE, channels=SD_CHANNELS, dtype=np.int16)
    sd.wait()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    import soundfile as sf
    sf.write(temp.name, audio, SD_RATE)
    return temp.name

def transcribe(file_path: str) -> str:
    """Send audio file to Whisper for STT"""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text.strip()

def speak(text: str) -> None:
    """Low-latency TTS: stream PCM chunks directly to speaker."""
    if not settings.eleven_api_key or not settings.eleven_voice_id:
        print("Ultron (text):", text)
        return

    client = ElevenLabs(api_key=settings.eleven_api_key)
    model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

    # Ask for raw PCM so we can stream immediately.
    # If your SDK doesn’t support this format, fall back below.
    result = client.text_to_speech.convert(
        voice_id=settings.eleven_voice_id,
        model_id=model_id,
        text=text,
        output_format="pcm_16000",   # 16kHz, 16-bit, mono PCM
    )

    # result may be bytes or an iterator of bytes; handle both.
    with sd.RawOutputStream(
        samplerate=SD_RATE, channels=SD_CHANNELS, dtype=SD_DTYPE, blocksize=0
    ) as out:
        if isinstance(result, (bytes, bytearray, memoryview)):
            out.write(bytes(result))
        else:
            for chunk in result:      # iterator of PCM chunks
                if not chunk:
                    continue
                out.write(chunk)      # stream straight to device
