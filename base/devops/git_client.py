
from __future__ import annotations
from typing import Optional, List
import os
import subprocess
from pathlib import Path
from loguru import logger

class GitError(RuntimeError):
    pass

class GitClient:
    """
    Small wrapper around git via subprocess to avoid forcing GitPython.
    Assumes `git` is installed and repo already initialized with a remote.
    """

    def __init__(self, repo_root: str | Path, remote: str = "origin"):
        self.root = Path(repo_root).resolve()
        self.remote = remote

    def _run(self, *args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
        logger.debug(f"[git] {' '.join(args)}")
        return subprocess.run(
            ["git", *args],
            cwd=str(self.root),
            check=check,
            capture_output=capture,
            text=True,
        )

    def current_branch(self) -> str:
        res = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return res.stdout.strip()

    def ensure_user(self, name: str, email: str) -> None:
        self._run("config", "user.name", name)
        self._run("config", "user.email", email)

    def fetch(self) -> None:
        self._run("fetch", self.remote, "--prune")

    def checkout(self, branch: str, create: bool = False, start_point: Optional[str] = None) -> None:
        if create:
            if start_point:
                self._run("checkout", "-b", branch, f"{self.remote}/{start_point}")
            else:
                self._run("checkout", "-b", branch)
        else:
            self._run("checkout", branch)

    def add_all(self, paths: Optional[List[str]] = None) -> None:
        if not paths:
            self._run("add", "-A")
        else:
            self._run("add", *paths)

    def commit(self, message: str) -> None:
        # Allow empty? Prefer not; create only when there are changes.
        try:
            self._run("commit", "-m", message)
        except subprocess.CalledProcessError as e:
            if "nothing to commit" in (e.stderr or ""):
                raise GitError("Nothing to commit")
            raise

    def push(self, branch: str, set_upstream: bool = True) -> None:
        if set_upstream:
            self._run("push", "-u", self.remote, branch)
        else:
            self._run("push", self.remote, branch)

    def has_changes(self) -> bool:
        res = self._run("status", "--porcelain")
        return bool(res.stdout.strip())
