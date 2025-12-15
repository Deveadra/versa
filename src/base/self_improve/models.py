# base/self_improve/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProposedChange:
    path: str
    apply_mode: str  # "replace_block" or "full_file"
    search_anchor: str | None
    replacement: str
    rationale: str | None = None


@dataclass
class Proposal:
    title: str
    description: str
    changes: list[ProposedChange]
