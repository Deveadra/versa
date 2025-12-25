# src/base/__init__.py
"""
Ultron MVP package.

Keep this module import-safe:
- Do NOT import optional/voice-heavy modules at import time.
- Expose modules lazily via __getattr__.
"""

__version__ = "0.1.0"
__all__: list[str] = ["__version__"]


def __getattr__(name: str):
    # Lazy imports so tests and non-voice installs don't explode.
    if name == "brain":
        from .llm import brain
        return brain

    if name == "core":
        from .core import core
        return core

    if name == "audio":
        from .core import audio
        return audio

    raise AttributeError(name)
