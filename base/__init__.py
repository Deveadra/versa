"""
Jarvis Offline Assistant Package


Modules:
- core: State manager, personalities, reset logic
- audio: Wake word detection, silence listening, playback
- brain: AI interaction, personality tuning
- plugins: Extensible plugin system (system stats, spotify, lights, calendar, email)
"""


__version__ = "0.1.0"


from .core import core
from .core import audio
from .llm import brain