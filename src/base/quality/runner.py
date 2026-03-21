from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from base.quality.models import (
    CommandRun,
    CommandSpec,
    Diagnostic,
    QualitySnapshot,
    Severity,
    ToolName,
)
from base.quality.policy import RepairPolicy


class QualityRunner:
    def __init__(self, repo_root: Path, policy: RepairPolicy) -> None:
        self.repo_root = repo_root.resolve()
        self.policy = policy

    def snapshot(
        self,
        files: tuple[Path, ...],
        *,
        run_typecheck: bool,
        run_tests: bool,
        pytest_args: tuple[str, ...] = (),
    ) -> QualitySnapshot:
        if not files:
            return QualitySnapshot()

        commands: list[CommandRun] = []
        diagnostics: list[Diagnostic] = []

        ruff_run, ruff_diagnostics = self.run_ruff_check(files, fix=False)
        commands.append(ruff_run)
        diagnostics.extend(ruff_diagnostics)

        if run_typecheck:
            pyright_run, pyright_diagnostics = self.run_pyright(files)
            commands.append(pyright_run)
            diagnostics.extend(pyright_diagnostics)

        if run_tests:
            pytest_run = self.run_pytest(files, pytest_args=pytest_args)
            commands.append(pytest_run)
            if not pytest_run.ok:
                diagnostics.append(
                    Diagnostic(
                        tool=ToolName.PYTEST,
                        code="PYTEST",
                        message="Pytest failed for the current repair scope.",
                        severity=Severity.ERROR,
                    )
                )

        return QualitySnapshot(
            diagnostics=tuple(diagnostics),
            commands=tuple(commands),
        )

    def run_ruff_format(self, files: tuple[Path, ...], *, fix: bool) -> CommandRun:
        args: list[str] = ["-m", self.policy.ruff_binary, "format"]
        if not fix:
            args.append("--check")
        args.extend(path.as_posix() for path in files)
        return self.run_command(
            CommandSpec(
                name=sys.executable,
                args=tuple(args),
                cwd=self.repo_root,
                description="ruff format",
            )
        )

    def run_ruff_check(
        self,
        files: tuple[Path, ...],
        *,
        fix: bool,
    ) -> tuple[CommandRun, tuple[Diagnostic, ...]]:
        args: list[str] = ["-m", self.policy.ruff_binary, "check", "--output-format", "json"]
        if fix:
            args.append("--fix")
        args.extend(path.as_posix() for path in files)
        command = self.run_command(
            CommandSpec(
                name=sys.executable,
                args=tuple(args),
                cwd=self.repo_root,
                description="ruff check",
            )
        )
        return command, self._parse_ruff_json(command.stdout)

    def run_pyright(
        self,
        files: tuple[Path, ...],
    ) -> tuple[CommandRun, tuple[Diagnostic, ...]]:
        args: list[str] = ["-m", self.policy.pyright_binary, "--outputjson"]
        args.extend(path.as_posix() for path in files)
        command = self.run_command(
            CommandSpec(
                name=sys.executable,
                args=tuple(args),
                cwd=self.repo_root,
                description="pyright",
            )
        )
        return command, self._parse_pyright_json(command.stdout)

    def run_pytest(
        self,
        files: tuple[Path, ...],
        *,
        pytest_args: tuple[str, ...],
    ) -> CommandRun:
        args: list[str] = ["-m", self.policy.pytest_binary]

        if pytest_args:
            args.extend(pytest_args)
        else:
            args.append("-q")

        test_targets = self._pytest_targets(files)
        args.extend(path.as_posix() for path in test_targets)

        return self.run_command(
            CommandSpec(
                name=sys.executable,
                args=tuple(args),
                cwd=self.repo_root,
                description="pytest",
            )
        )

    def run_command(self, spec: CommandSpec) -> CommandRun:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                spec.argv,
                cwd=spec.cwd or self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return CommandRun(
                spec=spec,
                returncode=completed.returncode,
                stdout=completed.stdout[-20000:],
                stderr=completed.stderr[-20000:],
                duration_seconds=time.perf_counter() - started,
            )
        except FileNotFoundError as exc:
            return CommandRun(
                spec=spec,
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_seconds=time.perf_counter() - started,
            )

    def _parse_ruff_json(self, stdout: str) -> tuple[Diagnostic, ...]:
        if not stdout.strip():
            return ()

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return (
                Diagnostic(
                    tool=ToolName.RUFF,
                    code="RUFF_PARSE",
                    message="Unable to parse Ruff JSON output.",
                    severity=Severity.ERROR,
                ),
            )

        diagnostics: list[Diagnostic] = []
        for item in payload:
            filename = item.get("filename")
            location = item.get("location") or {}
            end_location = item.get("end_location") or {}
            diagnostics.append(
                Diagnostic(
                    tool=ToolName.RUFF,
                    code=str(item.get("code") or "RUFF"),
                    message=str(item.get("message") or ""),
                    severity=Severity.ERROR,
                    path=self._safe_relpath(filename),
                    line=location.get("row"),
                    column=location.get("column"),
                    end_line=end_location.get("row"),
                    end_column=end_location.get("column"),
                    fixable=bool(item.get("fix")),
                )
            )
        return tuple(diagnostics)

    def _parse_pyright_json(self, stdout: str) -> tuple[Diagnostic, ...]:
        if not stdout.strip():
            return ()

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return (
                Diagnostic(
                    tool=ToolName.PYRIGHT,
                    code="PYRIGHT_PARSE",
                    message="Unable to parse Pyright JSON output.",
                    severity=Severity.ERROR,
                ),
            )

        diagnostics: list[Diagnostic] = []
        for item in payload.get("generalDiagnostics", []):
            file_path = item.get("file")
            range_data: dict[str, Any] = item.get("range") or {}
            start = range_data.get("start") or {}
            end = range_data.get("end") or {}
            severity_name = str(item.get("severity") or "error").lower()
            severity = Severity.WARNING if severity_name == "warning" else Severity.ERROR

            diagnostics.append(
                Diagnostic(
                    tool=ToolName.PYRIGHT,
                    code=str(item.get("rule") or "PYRIGHT"),
                    message=str(item.get("message") or ""),
                    severity=severity,
                    path=self._safe_relpath(file_path),
                    line=(start.get("line", 0) + 1) if isinstance(start.get("line"), int) else None,
                    column=(
                        (start.get("character", 0) + 1)
                        if isinstance(start.get("character"), int)
                        else None
                    ),
                    end_line=(end.get("line", 0) + 1) if isinstance(end.get("line"), int) else None,
                    end_column=(
                        (end.get("character", 0) + 1)
                        if isinstance(end.get("character"), int)
                        else None
                    ),
                )
            )
        return tuple(diagnostics)

    def _safe_relpath(self, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None

        try:
            return Path(raw_path).resolve().relative_to(self.repo_root)
        except ValueError:
            return Path(raw_path)

    @staticmethod
    def _pytest_targets(files: tuple[Path, ...]) -> tuple[Path, ...]:
        targets: list[Path] = []
        for path in files:
            posix = path.as_posix()
            name = path.name
            if posix.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py"):
                targets.append(path)
        return tuple(targets)
