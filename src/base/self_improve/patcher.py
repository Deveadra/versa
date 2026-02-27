from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass
class PatchOutcome:
    ok: bool
    message: str
    changed_files: list[str]

class Patcher:
    def __init__(self, repo_root: str, allowlist_prefixes: tuple[str, ...]) -> None:
        self.root = Path(repo_root)
        self.allowlist_prefixes = allowlist_prefixes

    def _allowed(self, relpath: str) -> bool:
        # conservative: only allow edits within known-safe trees
        target = (self.root / relpath).resolve()
        return any(target.is_relative_to(self.root / p) for p in self.allowlist_prefixes)

    def apply_anchor_replace(self, relpath: str, anchor: str, replacement: str) -> PatchOutcome:
        if not self._allowed(relpath):
            return PatchOutcome(False, f"Path not allowed: {relpath}", [])
        p = self.root / relpath
        if not p.exists():
            return PatchOutcome(False, f"File missing: {relpath}", [])
        text = p.read_text(encoding="utf-8", errors="ignore")
        if anchor not in text:
            return PatchOutcome(False, "Anchor not found (refuse full rewrite in janitor mode)", [])
        new_text = text.replace(anchor, replacement, 1)
        if new_text == text:
            return PatchOutcome(True, "No-op", [])
        p.write_text(new_text, encoding="utf-8")
        return PatchOutcome(True, "Replaced block", [relpath])
