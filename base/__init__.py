"""
Jarvis Offline Assistant Package


Modules:
- core: State manager, personalities, reset logic
- audio: Wake word detection, silence listening, playback
- brain: AI interaction, personality tuning
- plugins: Extensible plugin system (system stats, spotify, lights, calendar, email)
"""

__version__ = "0.1.0"

# Keep package imports lightweight to avoid pulling in optional runtime dependencies
# such as wake-word or TTS libraries during module import time.
from .core import core

__all__ = ["core"]
