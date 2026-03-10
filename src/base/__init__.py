# src/base/__init__.py

# from __future__ import annotations

# from .core import audio, core
# from .llm import brain

"""
Aerith MVP package.

Keep this module import-safe:
- Do NOT import optional/voice-heavy modules at import time.
- Expose modules lazily via __getattr__.
"""

__version__ = "0.1.0"
__all__: list[str] = ["__version__"]


# def __getattr__(name: str):
#     # Lazy imports so tests and non-voice installs don't explode.
#     if name == "brain":
#         return brain

#     if name == "core":
#         return core

#     if name == "audio":
#         return audio

#     raise AttributeError(name)
