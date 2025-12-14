import os
import struct
import tempfile
import threading
import time

import pvporcupine
import sounddevice as sd
import soundfile as sf
from playsound import playsound
from pvporcupine import create
from TTS.api import TTS

from .core import JarvisState, pick_ack, state, stop_playback

tts = TTS("tts_models/en/jenny/jenny")


def listen_until_silence(threshold=0.01, timeout=8, samplerate=16000):
    print("[Listening...]")
    wav_path = "input.wav"
    buffer = []
    silence_start = None

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(1024)
            buffer.append(data)
            vol = abs(data).mean()
            if vol < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > timeout:
                    break
            else:
                silence_start = None

    sf.write(wav_path, b"".join([d.tobytes() for d in buffer]), samplerate)
    return wav_path


def stream_speak(text):
    global stop_playback
    stop_playback = False

    def _play():
        global stop_playback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tts.tts_to_file(text=text, file_path=tmp.name)
            if not stop_playback:
                playsound(tmp.name)

    threading.Thread(target=_play, daemon=True).start()


def interrupt():
    global stop_playback
    stop_playback = True
    ack = pick_ack("stop")
    stream_speak(ack)


# 💡 print("DEBUG PICOVOICE KEY:", os.getenv("PICOVOICE_API_KEY"))


def listen_for_wake_word():
    """
    Initialize Porcupine wake word detection with error handling.
    Fails fast if the API key is missing or invalid.
    """
    access_key = os.getenv("PICOVOICE_API_KEY")

    # Check early for missing/empty key
    if not access_key:
        raise RuntimeError(
            "PICOVOICE_API_KEY is not loaded. "
            "Check your config/.env file. If your key contains '/' or '==', "
            "wrap it in quotes in the .env file like:\n\n"
            "PICOVOICE_API_KEY='your_key_here=='"
        )

    try:
        porcupine = create(
            access_key=access_key,
            # access_key=os.getenv("PICOVOICE_API_KEY"),
            keywords=["jarvis"],
        )
        print("[DEBUG] Porcupine initialized successfully.")

        pa = sd.RawInputStream(
            samplerate=porcupine.sample_rate,
            blocksize=porcupine.frame_length,
            channels=1,
            dtype="int16",
        )
        pa.start()
        print("[Idle... say 'Jarvis' to wake me up.]")
        while state == JarvisState.IDLE:
            pcm = pa.read(porcupine.frame_length)[0]
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                pa.stop()
                return
        return porcupine

    except pvporcupine.PorcupineActivationError as e:
        raise RuntimeError(
            f"Porcupine activation failed: {e}\n"
            "→ Check that your API key is valid and not expired."
        ) from e

    except pvporcupine.PorcupineActivationLimitError as e:
        raise RuntimeError(
            f"Porcupine activation limit reached: {e}\n"
            "→ You may need to reset or regenerate your key from the Picovoice Console."
        ) from e

    except Exception as e:
        raise RuntimeError(f"Unexpected error initializing Porcupine: {e}") from e
