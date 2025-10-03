# base/devops/git_client.py
from __future__ import annotations
from typing import Optional, List, Sequence, Union
import subprocess
from pathlib import Path
from loguru import logger

class GitError(RuntimeError):
    pass

class GitClient:
    """
    Small wrapper around git via subprocess to avoid GitPython dependency.
    Assumes `git` is installed and repo already initialized with a remote.
    """

    def __init__(self, repo_root: Union[str, Path], remote: str = "origin"):
        self.root = Path(repo_root).resolve()
        self.remote = remote

    def ensure_user(self, name: str, email: str) -> None:
        """
        Ensure Git has a user identity set for commits.
        Ensure git commits on this branch are attributed to Ultron (or whatever identity is passed).
        Does not overwrite your global identity unless explicitly run.
        """
        try:
            # Set local config only (repo scope, not global)
            self._run(["config", "--local", "user.name", name])
            self._run(["config", "--local", "user.email", email])
            logger.debug(f"Configured repo git user: {name} <{email}>")
        except GitError as e:
            logger.error(f"Failed to set git user identity: {e}")
            raise

    def _run(self, args: Sequence[str] | str, check: bool = True) -> str:
        # Normalize args -> flat list[str]
        if isinstance(args, str):
            args = args.split()
        else:
            flat: list[str] = []
            for a in args:
                if isinstance(a, (list, tuple)):
                    flat.extend(map(str, a))
                else:
                    flat.append(str(a))
            args = flat

        # Strip accidental leading "git" because we prepend it below
        if args and args[0].lower() == "git":
            args = args[1:]

        cmd = ["git"] + list(args)
        logger.debug(f"[git] {' '.join(cmd)}")

        res = subprocess.run(
            cmd,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out = res.stdout.strip()
        if check and res.returncode != 0:
            err = res.stderr.strip()
            raise GitError(err or out or "Unknown git error")
        return out

    def current_branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"])

    def has_uncommitted_changes(self) -> bool:
        return bool(self._run(["status", "--porcelain"], check=False).strip())

    def stash_push(self, message: str = "ultron-autosave") -> None:
        try:
            self._run(["stash", "push", "-u", "-m", message])
        except GitError as e:
            logger.warning(f"Stash failed (continuing): {e}")

    def stash_pop(self) -> None:
        try:
            self._run(["stash", "pop"])
        except GitError as e:
            # Not fatal — conflicts could occur; caller decides next steps.
            logger.warning(f"Stash pop failed (continuing): {e}")

    def fetch(self) -> None:
        self._run(["fetch", self.remote, "--prune"])

    def checkout(self, branch: str, create: bool = False, start_point: Optional[str] = None) -> None:
        try:
            if create:
                if start_point:
                    self._run(["checkout", "-b", branch, f"{self.remote}/{start_point}"])
                else:
                    self._run(["checkout", "-b", branch])
            else:
                self._run(["checkout", branch])
        except GitError as e:
            msg = str(e)
            if "already exists" in msg or "already on" in msg:
                # Reuse the branch if already there
                self._run(["checkout", branch], check=False)
            else:
                raise

    def safe_switch(self, target_branch: str, create: bool = False, start_point: Optional[str] = None) -> None:
        dirty = self.has_uncommitted_changes()
        if dirty:
            logger.info("Uncommitted changes detected — stashing")
            self.stash_push()

        try:
            self.checkout(target_branch, create=create, start_point=start_point)
        finally:
            if dirty:
                logger.info("Restoring stashed changes")
                self.stash_pop()

    def add_all(self, paths: Optional[List[str]] = None) -> None:
        if not paths:
            self._run(["add", "-A"])
        else:
            self._run(["add", *paths])

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
        return self.has_uncommitted_changes()
