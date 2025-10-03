# base/self_improve/models.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ProposedChange:
    path: str
    apply_mode: str           # "replace_block" or "full_file"
    search_anchor: Optional[str]
    replacement: str

@dataclass
class Proposal:
    title: str
    description: str
    changes: List[ProposedChange]
