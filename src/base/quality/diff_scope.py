from __future__ import annotations

import subprocess
from pathlib import Path

from base.quality.policy import RepairPolicy, RepairScopeMode


class GitDiffScopeResolver:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def resolve_files(
        self,
        policy: RepairPolicy,
        explicit_files: tuple[Path, ...] = (),
    ) -> tuple[Path, ...]:
        if policy.scope_mode is RepairScopeMode.EXPLICIT:
            return self._normalize_files(explicit_files, policy.file_extensions)
        if policy.scope_mode is RepairScopeMode.FULL_REPO:
            return self.repo_files(policy.file_extensions)
        if policy.scope_mode is RepairScopeMode.BRANCH_DELTA:
            return self.branch_delta_files(
                base_ref=policy.base_ref,
                head_ref=policy.head_ref,
                extensions=policy.file_extensions,
            )
        return self.changed_files(
            include_untracked=policy.include_untracked,
            extensions=policy.file_extensions,
        )

    def repo_files(self, extensions: tuple[str, ...]) -> tuple[Path, ...]:
        output = self._run_git("ls-files")
        files = tuple(Path(line) for line in output.splitlines() if line.strip())
        return self._normalize_files(files, extensions)

    def changed_files(
        self,
        include_untracked: bool,
        extensions: tuple[str, ...],
    ) -> tuple[Path, ...]:
        changed = set()

        diff_output = self._run_git("diff", "--name-only", "--diff-filter=ACMR", "HEAD")
        changed.update(line for line in diff_output.splitlines() if line.strip())

        staged_output = self._run_git(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
        )
        changed.update(line for line in staged_output.splitlines() if line.strip())

        if include_untracked:
            untracked_output = self._run_git(
                "ls-files",
                "--others",
                "--exclude-standard",
            )
            changed.update(line for line in untracked_output.splitlines() if line.strip())

        files = tuple(Path(value) for value in sorted(changed))
        return self._normalize_files(files, extensions)

    def branch_delta_files(
        self,
        base_ref: str,
        head_ref: str,
        extensions: tuple[str, ...],
    ) -> tuple[Path, ...]:
        output = self._run_git(
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}...{head_ref}",
        )
        files = tuple(Path(line) for line in output.splitlines() if line.strip())
        return self._normalize_files(files, extensions)

    def _run_git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    def _normalize_files(
        self,
        files: tuple[Path, ...],
        extensions: tuple[str, ...],
    ) -> tuple[Path, ...]:
        normalized: list[Path] = []
        for file_path in files:
            absolute = (self.repo_root / file_path).resolve()
            if not absolute.exists():
                continue
            if absolute.is_dir():
                continue
            if extensions and absolute.suffix not in extensions:
                continue
            normalized.append(absolute.relative_to(self.repo_root))
        return tuple(sorted(set(normalized)))
