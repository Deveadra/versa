# src/base/io/stream_loop.py
from __future__ import annotations

from .text_loop import run_text


def run_stream(orch):
    print("⚠️ Streaming STT/TTS not yet implemented. Falling back to text mode.")
    run_text(orch)
