# wherever interface.py lives

from config.config import settings  # not Settings

from .stream_loop import run_stream
from .text_loop import run_text
from .voice_loop import run_voice


def launch_interface(orch):
    mode = (settings.mode or "text").lower()
    if mode == "text":
        run_text(orch)
    elif mode == "voice":
        run_voice(orch)
    elif mode == "stream":
        run_stream(orch)
    else:
        raise ValueError(f"Unknown AERITH_MODE: {mode!r}")
