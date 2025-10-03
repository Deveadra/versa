from __future__ import annotations
from typing import Optional, List, Union
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

    def _run(self, args: Union[str, List[str]], check: bool = True, allow_warnings: bool = False) -> str:
        if isinstance(args, str):
            args = args.split()
        else:
            args = [str(a) for a in args]

        logger.debug(f"[git] {' '.join(args)}")

        result = subprocess.run(
            ["git"] + args,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            if allow_warnings and "warning:" in result.stderr.lower():
                logger.warning(f"Non-fatal git warning: {result.stderr.strip()}")
            elif check:
                raise GitError(result.stderr.strip() or "Unknown git error")

        return result.stdout.strip()


    def current_branch(self) -> str:
        """Return the currently checked-out branch name."""
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"])

    def has_uncommitted_changes(self) -> bool:
        """Check if the working directory has uncommitted or untracked changes."""
        return bool(self._run(["status", "--porcelain"]))

    def safe_switch(self, target_branch: str, create: bool = False):
        """
        Safely switch branches, stashing and restoring changes if needed.
        """
        dirty = self.has_uncommitted_changes()
        if dirty:
            logger.info("Uncommitted changes detected — stashing")
            self._run(["stash", "push", "-u", "-m", "ultron-autosave"])

        self.checkout(target_branch, create=create)

        if dirty:
            try:
                logger.info("Uncommitted changes detected — stashing")
                self._run(["stash", "push", "-u", "-m", "ultron-autosave"], allow_warnings=True)
            except Exception as e:
                logger.error(f"Failed to apply stashed changes: {e}")

    def ensure_user(self, name: str, email: str) -> None:
        self._run(["config", "user.name", name])
        self._run(["config", "user.email", email])

    def fetch(self) -> None:
        self._run(["fetch", self.remote, "--prune"])

    def checkout(self, branch: str, create: bool = False, start_point: Optional[str] = None) -> None:
        if create:
            if start_point:
                self._run(["checkout", "-b", branch, f"{self.remote}/{start_point}"])
            else:
                self._run(["checkout", "-b", branch])
        else:
            self._run(["checkout", branch])

    def add_all(self, paths: Optional[List[str]] = None) -> None:
        if not paths:
            self._run(["add", "-A"])
        else:
            self._run(["add"] + paths)

    def commit(self, message: str) -> None:
        try:
            self._run(["commit", "-m", message])
        except GitError as e:
            if "nothing to commit" in str(e):
                raise GitError("Nothing to commit")
            raise

    def push(self, branch: str, set_upstream: bool = True) -> None:
        if set_upstream:
            self._run(["push", "-u", self.remote, branch])
        else:
            self._run(["push", self.remote, branch])

    def has_changes(self) -> bool:
        return bool(self._run(["status", "--porcelain"]))
