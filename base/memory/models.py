
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

MemType = Literal["fact", "event", "device_log", "task"]

@dataclass
class MemoryItem:
    text: str
    type: MemType = "event"
    importance: float = 0.0