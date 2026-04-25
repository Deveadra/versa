from __future__ import annotations

import subprocess
from pathlib import Path

from base.quality.policy import RepairPolicy, RepairScopeMode


class GitDiffScopeResolver:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def resolve_candidate_files(
        self,
        policy: RepairPolicy,
        explicit_files: tuple[Path, ...] = (),
    ) -> tuple[Path, ...]:
        if policy.scope_mode is RepairScopeMode.EXPLICIT:
            return self._normalize_existing_files(explicit_files)
        if policy.scope_mode is RepairScopeMode.FULL_REPO:
            return self.repo_files()
        if policy.scope_mode is RepairScopeMode.BRANCH_DELTA:
            return self.branch_delta_files(
                base_ref=policy.base_ref,
                head_ref=policy.head_ref,
            )
        return self.changed_files(include_untracked=policy.include_untracked)

    def resolve_files(
        self,
        policy: RepairPolicy,
        explicit_files: tuple[Path, ...] = (),
    ) -> tuple[Path, ...]:
        candidates = self.resolve_candidate_files(policy, explicit_files=explicit_files)
        return self.filter_files(candidates, policy.file_extensions)

    def filter_files(
        self,
        files: tuple[Path, ...],
        extensions: tuple[str, ...],
    ) -> tuple[Path, ...]:
        if not extensions:
            return files

        normalized: list[Path] = []
        allowed = {extension.lower() for extension in extensions}
        for file_path in files:
            if file_path.suffix.lower() in allowed:
                normalized.append(file_path)
        return tuple(sorted(set(normalized)))

    def repo_files(self) -> tuple[Path, ...]:
        output = self._run_git("ls-files")
        files = tuple(Path(line) for line in output.splitlines() if line.strip())
        return self._normalize_existing_files(files)

    def changed_files(self, include_untracked: bool) -> tuple[Path, ...]:
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
        return self._normalize_existing_files(files)

    def branch_delta_files(
        self,
        base_ref: str,
        head_ref: str,
    ) -> tuple[Path, ...]:
        output = self._run_git(
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}...{head_ref}",
        )
        files = tuple(Path(line) for line in output.splitlines() if line.strip())
        return self._normalize_existing_files(files)

    def _run_git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    def _normalize_existing_files(self, files: tuple[Path, ...]) -> tuple[Path, ...]:
        normalized: list[Path] = []
        for file_path in files:
            absolute = (self.repo_root / file_path).resolve()
            if not absolute.exists():
                continue
            if absolute.is_dir():
                continue
            normalized.append(absolute.relative_to(self.repo_root))
        return tuple(sorted(set(normalized)))
