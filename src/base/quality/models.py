from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ToolName(StrEnum):
    RUFF = "ruff"
    PYRIGHT = "pyright"
    PYTEST = "pytest"
    SYSTEM = "system"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class Diagnostic:
    tool: ToolName
    code: str
    message: str
    severity: Severity = Severity.ERROR
    path: Path | None = None
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    fixable: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.path is not None:
            data["path"] = self.path.as_posix()
        data["tool"] = self.tool.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True)
class CommandSpec:
    name: str
    args: tuple[str, ...]
    cwd: Path | None = None
    description: str = ""

    @property
    def argv(self) -> tuple[str, ...]:
        return (self.name, *self.args)


@dataclass(slots=True)
class CommandRun:
    spec: CommandSpec
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "args": list(self.spec.args),
            "cwd": self.spec.cwd.as_posix() if self.spec.cwd is not None else None,
            "description": self.spec.description,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "ok": self.ok,
        }


@dataclass(slots=True)
class QualitySnapshot:
    diagnostics: tuple[Diagnostic, ...] = ()
    commands: tuple[CommandRun, ...] = ()

    @property
    def total_diagnostics(self) -> int:
        return len(self.diagnostics)

    @property
    def by_tool(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diagnostic in self.diagnostics:
            counts[diagnostic.tool.value] = counts.get(diagnostic.tool.value, 0) + 1
        return counts

    @property
    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diagnostic in self.diagnostics:
            counts[diagnostic.code] = counts.get(diagnostic.code, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_diagnostics": self.total_diagnostics,
            "by_tool": self.by_tool,
            "by_rule": self.by_rule,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass(slots=True)
class FileRepair:
    path: Path
    changed: bool
    rules_applied: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "changed": self.changed,
            "rules_applied": list(self.rules_applied),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class RepairRound:
    number: int
    repairs: tuple[FileRepair, ...]
    snapshot_before: QualitySnapshot
    snapshot_after: QualitySnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "repairs": [repair.to_dict() for repair in self.repairs],
            "snapshot_before": self.snapshot_before.to_dict(),
            "snapshot_after": self.snapshot_after.to_dict(),
        }


@dataclass(slots=True)
class RepairReport:
    repo_root: Path
    scope_name: str
    files_in_scope: tuple[Path, ...]
    baseline: QualitySnapshot
    after_autofix: QualitySnapshot
    final_snapshot: QualitySnapshot
    rounds: tuple[RepairRound, ...] = ()
    blocked: tuple[Diagnostic, ...] = ()
    notes: tuple[str, ...] = ()
    report_dir: Path | None = field(default=None)

    @property
    def success(self) -> bool:
        return self.final_snapshot.total_diagnostics == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root.as_posix(),
            "scope_name": self.scope_name,
            "files_in_scope": [path.as_posix() for path in self.files_in_scope],
            "baseline": self.baseline.to_dict(),
            "after_autofix": self.after_autofix.to_dict(),
            "final_snapshot": self.final_snapshot.to_dict(),
            "rounds": [round_.to_dict() for round_ in self.rounds],
            "blocked": [diagnostic.to_dict() for diagnostic in self.blocked],
            "notes": list(self.notes),
            "report_dir": self.report_dir.as_posix() if self.report_dir is not None else None,
            "success": self.success,
        }
