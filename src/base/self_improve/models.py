# base/self_improve/models.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProposedChange:
    path: str
    apply_mode: str  # "replace_block" or "full_file"
    search_anchor: str | None
    replacement: str


@dataclass
class Proposal:
    title: str
    description: str
    changes: list[ProposedChange]
