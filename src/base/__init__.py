# src/base/__init__.py
"""
Jarvis Offline Assistant Package


Modules:
- core: State manager, personalities, reset logic
- audio: Wake word detection, silence listening, playback
- brain: AI interaction, personality tuning
- plugins: Extensible plugin system (system stats, spotify, lights, calendar, email)
"""

__version__ = "0.1.0"
__all__: list[str] = []

from .core import audio, core
from .llm import brain


def __getattr__(name: str):
    if name == "brain":
        from .llm import brain  # imported only when actually accessed
        return brain
    raise AttributeError(name)