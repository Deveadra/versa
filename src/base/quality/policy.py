from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

PYTHON_EXTENSIONS: tuple[str, ...] = (".py",)
TYPESCRIPT_EXTENSIONS: tuple[str, ...] = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
)
REPAIRABLE_EXTENSIONS: tuple[str, ...] = PYTHON_EXTENSIONS + TYPESCRIPT_EXTENSIONS


class RepairScopeMode(StrEnum):
    CHANGED_FILES = "changed_files"
    BRANCH_DELTA = "branch_delta"
    FULL_REPO = "full_repo"
    EXPLICIT = "explicit"


@dataclass(slots=True)
class RepairPolicy:
    scope_mode: RepairScopeMode = RepairScopeMode.CHANGED_FILES
    base_ref: str = "main"
    head_ref: str = "HEAD"
    include_untracked: bool = True
    file_extensions: tuple[str, ...] = REPAIRABLE_EXTENSIONS

    auto_format: bool = True
    auto_fix_ruff: bool = True
    run_typecheck: bool = True
    run_tests: bool = False
    pytest_args: tuple[str, ...] = ()

    max_rounds: int = 3
    max_repairs_per_round: int = 25
    fail_on_unresolved: bool = True
    fail_on_unsupported_files: bool = False

    report_root: Path = Path("artifacts/quality")
    allowed_rule_codes: frozenset[str] = field(default_factory=lambda: frozenset({"B904", "E722"}))

    ruff_binary: str = "ruff"
    pyright_binary: str = "pyright"
    pytest_binary: str = "pytest"
    pnpm_binary: str = "pnpm"

    @property
    def python_extensions(self) -> tuple[str, ...]:
        return PYTHON_EXTENSIONS

    @property
    def typescript_extensions(self) -> tuple[str, ...]:
        return TYPESCRIPT_EXTENSIONS

    @classmethod
    def for_changed_files(cls) -> "RepairPolicy":
        return cls(scope_mode=RepairScopeMode.CHANGED_FILES)

    @classmethod
    def for_branch_delta(cls, base_ref: str, head_ref: str = "HEAD") -> "RepairPolicy":
        return cls(
            scope_mode=RepairScopeMode.BRANCH_DELTA,
            base_ref=base_ref,
            head_ref=head_ref,
        )

    @classmethod
    def for_full_repo(cls) -> "RepairPolicy":
        return cls(scope_mode=RepairScopeMode.FULL_REPO, include_untracked=False)
