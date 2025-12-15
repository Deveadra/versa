# base/voice/FastAPI.py
import base64
import io

import soundfile as sf
from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
from TTS.api import TTS

app = FastAPI()
tts = TTS(model_path="path/to/ultron_model", gpu=True)


class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
def speak(req: SpeakRequest):
    # synthesize to memory
    wav = tts.tts(req.text)
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=22050, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"audio": audio_b64}
