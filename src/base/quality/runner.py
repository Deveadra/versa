from __future__ import annotations

import errno
import json
import re
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
from base.quality.tooling import resolve_binary

TSC_PATTERN = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<column>\d+)\): "
    r"(?P<severity>error|warning) (?P<code>TS\d+): (?P<message>.*)$"
)


class QualityRunner:
    def __init__(self, repo_root: Path, policy: RepairPolicy) -> None:
        self.repo_root = repo_root.resolve()
        self.policy = policy

    def _resolve_pnpm(self) -> str | None:
        return resolve_binary(self.repo_root, self.policy.pnpm_binary, "pnpm", "corepack")

    def _resolve_node_tool(self, tool_name: str) -> str | None:
        return resolve_binary(self.repo_root, tool_name)

    def _tool_unavailable_diagnostic(self, code: str, message: str) -> tuple[Diagnostic, ...]:
        return (
            Diagnostic(
                tool=ToolName.SYSTEM,
                code=code,
                message=message,
                severity=Severity.ERROR,
            ),
        )

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

        python_files = self._files_with_suffixes(files, self.policy.python_extensions)
        typescript_files = self._files_with_suffixes(files, self.policy.typescript_extensions)

        if python_files:
            ruff_run, ruff_diagnostics = self.run_ruff_check(python_files, fix=False)
            commands.append(ruff_run)
            diagnostics.extend(ruff_diagnostics)

            if run_typecheck:
                pyright_run, pyright_diagnostics = self.run_pyright(python_files)
                commands.append(pyright_run)
                diagnostics.extend(pyright_diagnostics)

        if typescript_files:
            prettier_run, prettier_diagnostics = self.run_prettier_check(typescript_files)
            commands.append(prettier_run)
            diagnostics.extend(prettier_diagnostics)

            if run_typecheck:
                for package_root in self._typescript_package_roots(typescript_files):
                    tsc_run, tsc_diagnostics = self.run_tsc(package_root)
                    commands.append(tsc_run)
                    diagnostics.extend(tsc_diagnostics)

        if run_tests and python_files:
            pytest_run = self.run_pytest(python_files, pytest_args=pytest_args)
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

    def apply_autofix(self, files: tuple[Path, ...]) -> None:
        python_files = self._files_with_suffixes(files, self.policy.python_extensions)
        typescript_files = self._files_with_suffixes(files, self.policy.typescript_extensions)

        if python_files:
            if self.policy.auto_format:
                self.run_ruff_format(python_files, fix=True)
            if self.policy.auto_fix_ruff:
                self.run_ruff_check(python_files, fix=True)

        if typescript_files and self.policy.auto_format:
            self.run_prettier_write(typescript_files)

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
        args: list[str] = ["--outputjson"]
        args.extend(path.as_posix() for path in files)
        command = self.run_command(
            CommandSpec(
                name=self.policy.pyright_binary,
                args=tuple(args),
                cwd=self.repo_root,
                description="pyright",
            )
        )
        return command, self._parse_pyright_json(command.stdout)

    def run_prettier_check(
        self,
        files: tuple[Path, ...],
    ) -> tuple[CommandRun, tuple[Diagnostic, ...]]:
        if prettier := self._resolve_node_tool("prettier"):
            command = self.run_command(
                CommandSpec(
                    name=prettier,
                    args=("--check", *(path.as_posix() for path in files)),
                    cwd=self.repo_root,
                    description="prettier check",
                )
            )
            return command, self._parse_prettier_output(command, files)

        if pnpm := self._resolve_pnpm():
            argv = ("exec", "prettier", "--check", *(path.as_posix() for path in files))
            command = self.run_command(
                CommandSpec(
                    name=pnpm,
                    args=argv,
                    cwd=self.repo_root,
                    description="prettier check",
                )
            )
            return command, self._parse_prettier_output(command, files)

        command = CommandRun(
            spec=CommandSpec(
                name="prettier",
                args=(),
                cwd=self.repo_root,
                description="prettier check",
            ),
            returncode=127,
            stdout="",
            stderr="Prettier not found in node_modules/.bin or PATH.",
            duration_seconds=0.0,
        )
        return command, self._tool_unavailable_diagnostic(
            "TOOL_UNAVAILABLE",
            "Prettier is not available to the repair runner.",
        )

    def run_prettier_write(self, files: tuple[Path, ...]) -> CommandRun:
        if prettier := self._resolve_node_tool("prettier"):
            return self.run_command(
                CommandSpec(
                    name=prettier,
                    args=("--write", *(path.as_posix() for path in files)),
                    cwd=self.repo_root,
                    description="prettier write",
                )
            )

        pnpm = self._resolve_pnpm()
        if pnpm:
            return self.run_command(
                CommandSpec(
                    name=pnpm,
                    args=("exec", "prettier", "--write", *(path.as_posix() for path in files)),
                    cwd=self.repo_root,
                    description="prettier write",
                )
            )

        return CommandRun(
            spec=CommandSpec(
                name="prettier",
                args=(),
                cwd=self.repo_root,
                description="prettier write",
            ),
            returncode=127,
            stdout="",
            stderr="Prettier not found in node_modules/.bin or PATH.",
            duration_seconds=0.0,
        )

    def run_tsc(
        self,
        package_root: Path,
    ) -> tuple[CommandRun, tuple[Diagnostic, ...]]:
        if tsc := self._resolve_node_tool("tsc"):
            command = self.run_command(
                CommandSpec(
                    name=tsc,
                    args=("-p", "tsconfig.json", "--noEmit", "--pretty", "false"),
                    cwd=self.repo_root / package_root,
                    description=f"tsc ({package_root.as_posix()})",
                )
            )
            return command, self._parse_tsc_output(command.stdout)

        if pnpm := self._resolve_pnpm():
            command = self.run_command(
                CommandSpec(
                    name=pnpm,
                    args=("exec", "tsc", "-p", "tsconfig.json", "--noEmit", "--pretty", "false"),
                    cwd=self.repo_root / package_root,
                    description=f"tsc ({package_root.as_posix()})",
                )
            )
            return command, self._parse_tsc_output(command.stdout)

        command = CommandRun(
            spec=CommandSpec(
                name="tsc",
                args=(),
                cwd=self.repo_root / package_root,
                description=f"tsc ({package_root.as_posix()})",
            ),
            returncode=127,
            stdout="",
            stderr="TypeScript compiler not found in node_modules/.bin or PATH.",
            duration_seconds=0.0,
        )
        return command, self._tool_unavailable_diagnostic(
            "TOOL_UNAVAILABLE",
            f"TypeScript compiler is not available for {package_root.as_posix()}.",
        )

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
        except OSError as exc:
            returncode = 127 if exc.errno == errno.ENOENT else 126

            return CommandRun(
                spec=spec,
                returncode=returncode,
                stdout="",
                stderr=f"{exc.__class__.__name__}: {exc}",
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

    def _parse_prettier_output(
        self,
        command: CommandRun,
        files: tuple[Path, ...],
    ) -> tuple[Diagnostic, ...]:
        if command.ok:
            return ()

        diagnostics: list[Diagnostic] = []
        known_files = {path.as_posix(): path for path in files}

        for line in (command.stdout + "\n" + command.stderr).splitlines():
            stripped = line.strip()
            if not stripped.startswith("[warn] "):
                continue

            candidate = stripped.removeprefix("[warn] ").strip()
            if candidate in known_files:
                diagnostics.append(
                    Diagnostic(
                        tool=ToolName.SYSTEM,
                        code="PRETTIER",
                        message="Prettier formatting check failed.",
                        severity=Severity.ERROR,
                        path=known_files[candidate],
                    )
                )

        if diagnostics:
            return tuple(diagnostics)

        return (
            Diagnostic(
                tool=ToolName.SYSTEM,
                code="PRETTIER",
                message="Prettier formatting check failed.",
                severity=Severity.ERROR,
            ),
        )

    def _parse_tsc_output(self, stdout: str) -> tuple[Diagnostic, ...]:
        if not stdout.strip():
            return ()

        diagnostics: list[Diagnostic] = []
        for line in stdout.splitlines():
            match = TSC_PATTERN.match(line.strip())
            if not match:
                continue

            severity_name = match.group("severity").lower()
            severity = Severity.WARNING if severity_name == "warning" else Severity.ERROR

            diagnostics.append(
                Diagnostic(
                    tool=ToolName.SYSTEM,
                    code=match.group("code"),
                    message=match.group("message"),
                    severity=severity,
                    path=self._safe_relpath(match.group("file")),
                    line=int(match.group("line")),
                    column=int(match.group("column")),
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

    @staticmethod
    def _files_with_suffixes(
        files: tuple[Path, ...],
        suffixes: tuple[str, ...],
    ) -> tuple[Path, ...]:
        allowed = {suffix.lower() for suffix in suffixes}
        return tuple(path for path in files if path.suffix.lower() in allowed)

    def _typescript_package_roots(self, files: tuple[Path, ...]) -> tuple[Path, ...]:
        package_roots: set[Path] = set()

        for path in files:
            absolute = (self.repo_root / path).resolve()
            current = absolute.parent

            while current not in (self.repo_root, current.parent):
                if (current / "package.json").exists() and (current / "tsconfig.json").exists():
                    package_roots.add(current.relative_to(self.repo_root))
                    break
                current = current.parent

        return tuple(sorted(package_roots))
