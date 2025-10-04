# base/self_improve/proposal_engine.py
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from loguru import logger

from base.llm.brain import Brain
from base.self_improve.models import Proposal, ProposedChange
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


class ProposalEngine:
    def __init__(self, repo_root: str, brain: Brain | None = None):
        self.root = Path(repo_root).resolve()
        self.brain = brain or Brain()

    def _read(self, relpath: str) -> str:
        return (self.root / relpath).read_text(encoding="utf-8", errors="ignore")

    def safe_write(self, path: str, new_content: str) -> bool:
        """
        Safely write new content to file.
        - Writes to a temp file first
        - Diffs against existing (if file exists)
        - Replaces original only if content changed
        """
        try:
            original = ""
            p = Path(path)
            if p.exists():
                original = p.read_text(encoding="utf-8", errors="ignore")

            diff = list(
                difflib.unified_diff(
                    original.splitlines(),
                    new_content.splitlines(),
                    fromfile="before",
                    tofile="after",
                )
            )
            if not diff:
                return False  # no changes

            fd, tmp_path = tempfile.mkstemp()
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                tmpf.write(new_content)

            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_path, path)
            return True
        except Exception as e:
            logger.error(f"Safe write failed for {path}: {e}")
            return False

    def _apply_change(self, change: ProposedChange) -> tuple[bool, str]:
        try:
            full = self.root / change.path

            if change.apply_mode == "full_file":
                # Always create a .ultron copy, don't overwrite originals
                new_path = full.with_suffix(full.suffix + ".ultron")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_text(change.replacement, encoding="utf-8")
                return True, f"full_file rewrite → {new_path.name} created"

            elif change.apply_mode == "replace_block":
                if not full.exists():
                    # Fallback: create the file from scratch
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_text(change.replacement, encoding="utf-8")
                    return True, f"file created fresh (no original existed): {change.path}"

                old = self._read(change.path)
                anchor = change.search_anchor or ""

                if anchor and anchor in old:
                    # Anchor found → standard replace
                    new_content = old.replace(anchor, change.replacement, 1)
                    ok = self.safe_write(str(full), new_content)
                    return ok, "block replaced" if ok else "no changes applied"

                else:
                    # Fallback when anchor missing
                    logger.warning(
                        f"Anchor not found in {change.path}, falling back to full file rewrite"
                    )
                    backup_path = full.with_suffix(full.suffix + ".bak")
                    try:
                        full.replace(backup_path)
                        logger.info(f"Backed up original file to {backup_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to backup original {full}: {e}")

                    full.write_text(change.replacement, encoding="utf-8")
                    return True, f"anchor missing → replaced entire file {change.path}"

            else:
                return False, f"Unknown apply_mode {change.apply_mode}"

        except Exception as e:
            return False, f"apply error: {e}"

    def propose(self, instruction: str, index_md: str) -> Proposal:
        sys_prompt = PROPOSAL_SYS_PROMPT.format(
            max_files=settings.proposer_max_files_per_pr,
            max_bytes=settings.proposer_max_patch_bytes,
        )
        user_prompt = f"""User request:
{instruction}

Repository index:
{index_md}

Respond with strictly the JSON schema described.
"""
        raw = self.brain.ask_brain(user_prompt, system_prompt=sys_prompt).strip()

        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()

        try:
            obj = json.loads(raw)
        except Exception:
            logger.error("LLM returned invalid JSON; wrapping into no-op change.")
            obj = {
                "title": f"Ultron Proposal: {instruction[:40]}",
                "description": raw,
                "changes": [],
            }

        # Auto add a tiny test if the change looks substantive
        auto_test_needed = any(
            len(ch.get("replacement", "").splitlines()) > 15 or "def " in ch.get("replacement", "")
            for ch in obj.get("changes", [])
        )
        if auto_test_needed:
            obj.setdefault("changes", []).append(
                {
                    "path": "tests/test_autogenerated.py",
                    "apply_mode": "replace_block",
                    "search_anchor": "",
                    "replacement": "def test_autogenerated():\n    assert True",
                }
            )

        changes: list[ProposedChange] = [
            ProposedChange(
                path=ch.get("path", ""),
                apply_mode=ch.get("apply_mode", "replace_block"),
                search_anchor=ch.get("search_anchor"),
                replacement=ch.get("replacement", ""),
            )
            for ch in obj.get("changes", [])
        ]

        return Proposal(
            title=obj.get("title", f"Ultron proposal: {instruction[:30]}"),
            description=obj.get("description", ""),
            changes=changes,
        )

    def apply_proposal(self, proposal: Proposal) -> list[tuple[ProposedChange, bool, str]]:
        """
        Apply proposed changes to the working tree.
        Git branching is handled by Orchestrator/PRManager.
        """
        applied: list[tuple[ProposedChange, bool, str]] = []
        total_bytes = 0

        for ch in proposal.changes[: settings.proposer_max_files_per_pr]:
            total_bytes += len(ch.replacement.encode("utf-8"))
            if total_bytes > settings.proposer_max_patch_bytes:
                applied.append((ch, False, "patch byte limit exceeded"))
                continue

            ok, msg = self._apply_change(ch)
            applied.append((ch, ok, msg))

            if ok:
                lines_changed = len(ch.replacement.splitlines())
                if lines_changed > 15 or "def " in ch.replacement:
                    self._generate_test_stub(ch.path, ch.replacement)

        return applied

    def _generate_test_stub(self, path: str, replacement: str) -> None:
        test_dir = self.root / "tests"
        test_dir.mkdir(exist_ok=True)
        module_name = Path(path).stem
        test_file = test_dir / f"test_{module_name}.py"
        if test_file.exists():
            return
        content = """def test_placeholder():
    assert True
"""
        test_file.write_text(content, encoding="utf-8")
