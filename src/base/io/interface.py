# wherever interface.py lives
from config.config import settings  # not Settings


def launch_interface(orch):
    mode = (settings.mode or "text").lower()
    if mode == "text":
        from .text_loop import run_text

        run_text(orch)
    elif mode == "voice":
        from .voice_loop import run_voice

        run_voice(orch)
    elif mode == "stream":
        from .stream_loop import run_stream

        run_stream(orch)
    else:
        raise ValueError(f"Unknown ULTRON_MODE: {mode!r}")
