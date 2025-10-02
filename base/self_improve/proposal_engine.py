
from __future__ import annotations
from dataclasses import dataclass
import difflib
from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches
import re
from pathlib import Path
from loguru import logger
from typing import Tuple

from base.self_improve.code_indexer import CodeIndexer
from base.llm.brain import Brain
from config.config import settings

PROPOSAL_SYS_PROMPT = """You are Ultron's code proposer.
Given the user's natural-language request and a summary of the repository,
propose a *minimal, safe patch set*.

Return a JSON object with:
{{
  "title": "...",
  "description": "...",
  "changes": [
    {{
      "path": "relative/path.py",
      "apply_mode": "replace_block" | "full_file",
      "search_anchor": "...",
      "replacement": "..."
    }},
    ...
  ]
}}

Rules:
- Only modify files within the allowlist.
- Keep changes small; do not exceed {max_files} files or {max_bytes} bytes total replacement.
- Prefer 'replace_block' with a reliable 'search_anchor' snippet to find the place to change.
- NEVER include secrets or tokens.
"""

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

class ProposalEngine:
    def __init__(self, repo_root: str, brain: Optional[Brain] = None):
        self.root = Path(repo_root).resolve()
        self.brain = brain or Brain()

    def _read(self, relpath: str) -> str:
        return (self.root / relpath).read_text(encoding="utf-8", errors="ignore")

    def _write(self, relpath: str, content: str) -> None:
        p = self.root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def _apply_change(self, change: ProposedChange) -> Tuple[bool, str]:
        try:
            full = self.root / change.path

            # If file doesn't exist and it's not a full file replacement
            if not full.exists() and change.apply_mode != "full_file":
                return False, f"Target {change.path} not found for anchor replace"

            if change.apply_mode == "full_file":
                # Overwrite entire file
                self._write(change.path, change.replacement)
                return True, "full_file replaced"

            elif change.apply_mode == "replace_block":
                old = self._read(change.path)
                anchor = change.search_anchor or ""

                if anchor not in old:
                    # Try fuzzy match
                    close = get_close_matches(anchor, old.splitlines(), n=1, cutoff=0.6)
                    if close:
                        anchor = close[0]
                    else:
                        return False, f"Anchor not found in {change.path}"

                new_content = old.replace(anchor, change.replacement, 1)
                self._write(change.path, new_content)
                return True, "block replaced"

            else:
                return False, f"Unknown apply_mode {change.apply_mode}"

        except Exception as e:
            return False, f"apply error: {e}"


    def propose(self, instruction: str, index_md: str) -> Proposal:
        sys_prompt = PROPOSAL_SYS_PROMPT.format(
            max_files=settings.proposer_max_files_per_pr,
            max_bytes=settings.proposer_max_patch_bytes
        )
        user_prompt = f"""User request:
    {instruction}

    Repository index:
    {index_md}

    Respond with strictly the JSON schema described.
    """

        # Call LLM
        raw = self.brain.ask_brain(user_prompt, system_prompt=sys_prompt)

        # Clean wrapper code fences if the model wrapped JSON in ```json ... ```
        if raw and raw.startswith("```"):
            import re
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
            raw = raw.rstrip("`").strip()

        # Defensive parse
        import json
        try:
            obj = json.loads(raw)
        except Exception:
            logger.error("LLM returned invalid JSON; wrapping into a single no-op change.")
            obj = {"title": "Ultron proposal", "description": raw, "changes": []}

        changes: List[ProposedChange] = []
        for ch in obj.get("changes", []):
            changes.append(
                ProposedChange(
                    path=ch.get("path", ""),
                    apply_mode=ch.get("apply_mode", "replace_block"),
                    search_anchor=ch.get("search_anchor"),
                    replacement=ch.get("replacement", ""),
                )
            )

        return Proposal(
            title=obj.get("title", "Ultron proposal"),
            description=obj.get("description", ""),
            changes=changes,
        )

    def apply_proposal(self, proposal: Proposal) -> List[Tuple[ProposedChange, bool, str]]:
        applied = []
        total_bytes = 0
        for ch in proposal.changes[: settings.proposer_max_files_per_pr]:
            total_bytes += len(ch.replacement.encode("utf-8"))
            if total_bytes > settings.proposer_max_patch_bytes:
                applied.append((ch, False, "patch byte limit exceeded"))
                continue
            ok, msg = self._apply_change(ch)
            applied.append((ch, ok, msg))
        return applied
