from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from base.quality.models import Diagnostic, FileRepair


class SafeStrategyApplier:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def apply(
        self,
        diagnostics: tuple[Diagnostic, ...],
        *,
        max_repairs: int,
    ) -> tuple[FileRepair, ...]:
        grouped: dict[Path, list[Diagnostic]] = defaultdict(list)
        for diagnostic in diagnostics[:max_repairs]:
            if diagnostic.path is None:
                continue
            grouped[diagnostic.path].append(diagnostic)

        repairs: list[FileRepair] = []
        for relative_path, file_diagnostics in grouped.items():
            repairs.append(self._apply_file_repairs(relative_path, file_diagnostics))
        return tuple(repairs)

    def _apply_file_repairs(
        self,
        relative_path: Path,
        diagnostics: list[Diagnostic],
    ) -> FileRepair:
        absolute_path = self.repo_root / relative_path
        original_text = absolute_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)

        rules_applied: list[str] = []
        notes: list[str] = []

        for diagnostic in diagnostics:
            if diagnostic.code == "E722":
                if self._apply_e722(lines, diagnostic):
                    rules_applied.append("E722")
            elif diagnostic.code == "B904":
                result = self._apply_b904(lines, diagnostic)
                if result == "changed":
                    rules_applied.append("B904")
                elif result == "skipped":
                    notes.append(
                        f"Skipped B904 at line {diagnostic.line}; could not safely infer exception source."
                    )

        new_text = "".join(lines)
        changed = new_text != original_text
        if changed:
            absolute_path.write_text(new_text, encoding="utf-8")

        return FileRepair(
            path=relative_path,
            changed=changed,
            rules_applied=tuple(sorted(set(rules_applied))),
            notes=tuple(notes),
        )

    def _apply_e722(self, lines: list[str], diagnostic: Diagnostic) -> bool:
        if diagnostic.line is None:
            return False
        index = diagnostic.line - 1
        if index < 0 or index >= len(lines):
            return False

        original = lines[index]
        updated = re.sub(r"^(\s*)except\s*:\s*$", r"\1except Exception:\n", original)
        if updated == original:
            return False

        lines[index] = updated
        return True

    def _apply_b904(self, lines: list[str], diagnostic: Diagnostic) -> str:
        if diagnostic.line is None:
            return "skipped"

        index = diagnostic.line - 1
        if index < 0 or index >= len(lines):
            return "skipped"

        raise_line = lines[index]
        stripped = raise_line.strip()

        if not stripped.startswith("raise "):
            return "skipped"
        if " from " in stripped:
            return "skipped"

        alias = self._find_exception_alias(lines, index)
        if alias is None:
            return "skipped"

        newline = "\n" if raise_line.endswith("\n") else ""
        line_without_newline = raise_line[:-1] if newline else raise_line
        lines[index] = f"{line_without_newline} from {alias}{newline}"
        return "changed"

    def _find_exception_alias(self, lines: list[str], raise_index: int) -> str | None:
        raise_indent = self._indent_of(lines[raise_index])

        for index in range(raise_index - 1, -1, -1):
            stripped = lines[index].strip()
            if not stripped:
                continue

            indent = self._indent_of(lines[index])
            if indent > raise_indent:
                continue

            match = re.match(
                r"^except\b.*\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$",
                stripped,
            )
            if match:
                return match.group(1)

            if stripped.startswith(
                (
                    "except:",
                    "def ",
                    "class ",
                    "for ",
                    "while ",
                    "if ",
                    "with ",
                    "try:",
                )
            ):
                break

        return None

    @staticmethod
    def _indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))
