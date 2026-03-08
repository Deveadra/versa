# base/self_improve/pr_manager.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests
from loguru import logger

from base.devops.git_client import GitClient, GitError
from base.self_improve.models import Proposal
from config.config import settings


class PRManager:
    """
    Creates a branch, commits applied changes, pushes, opens a PR via GitHub API.
    """

    def __init__(self, repo_root: str, repo_slug: str | None = None, token: str | None = None):
        self.root = Path(repo_root).resolve()
        self.repo_slug = repo_slug or settings.github_repo
        self.token = token or settings.github_token
        self.client = GitClient(self.root, remote=settings.github_remote_name)
        # Back-compat aliases (older code uses these names)
        self.git = self.client
        self.logger = logger.bind(component="PRManager", repo=str(self.root))
        self.original_branch: str | None = None
        self._pending_stash_pop: bool = False
        self._autosave_stash_ref: str | None = None
        self._autosave_stash_sha: str | None = None

    def _github_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
        }
        api_version = getattr(settings, "github_api_version", None)
        if api_version:
            headers["X-GitHub-Api-Version"] = str(api_version)

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def _github_request(self, method: str, url: str, **kwargs) -> requests.Response:
        timeout = int(getattr(settings, "github_api_timeout_sec", 20))
        headers = kwargs.pop("headers", {}) or {}
        merged_headers = self._github_headers()
        merged_headers.update(headers)
        return requests.request(
            method,
            url,
            headers=merged_headers,
            timeout=timeout,
            **kwargs,
        )

    def _github_ready(self) -> bool:
        if not self.repo_slug:
            self.logger.error("GitHub repo slug is not configured.")
            return False
        if not self.token:
            self.logger.error("GitHub token is not configured.")
            return False
        return True

    def _repo_owner(self) -> str:
        if not self.repo_slug or "/" not in self.repo_slug:
            raise ValueError(f"Invalid github repo slug: {self.repo_slug!r}")
        return self.repo_slug.split("/", 1)[0]

    def prepare_branch(
        self, branch_name: str, base: str = "main", *, restore_stash: bool = True
    ) -> str:
        """
        Create/switch to a fresh proposal branch safely.

        - Stash uncommitted local changes (if any).
        - Preserve the first autosave restore context across nested prepare_branch() calls.
        - If branch already exists locally or remotely, append a short suffix.
        - Ensures user.name/email are set for the repo.

        Returns the final branch name.
        """
        self.git.ensure_user(
            getattr(settings, "github_bot_name", "aerith-bot"),
            getattr(settings, "github_bot_email", "aerith-bot@local"),
        )

        try:
            current_branch = self.git.current_branch()
        except Exception:
            current_branch = None

        # Preserve the *user's* original branch across nested prepare_branch() calls.
        if self.original_branch is None and current_branch:
            self.original_branch = current_branch

        final = branch_name
        suffix = 0
        while self.git.branch_exists(final):
            suffix += 1
            final = f"{branch_name}-{suffix}"

        pending_restore = bool(
            getattr(self, "_pending_stash_pop", False)
            or getattr(self, "_autosave_stash_ref", None)
            or getattr(self, "_autosave_stash_sha", None)
        )

        stashed_this_call = False
        stash_ref: str | None = None
        stash_sha: str | None = None

        if self.git.has_uncommitted_changes():
            self.logger.info("Uncommitted changes detected — stashing")
            stash_ref, stash_sha = self.git.stash_push("aerith-autosave")
            stashed_this_call = bool(stash_ref or stash_sha)

            if stashed_this_call:
                # Only record a new restore target if one is not already being tracked.
                if pending_restore:
                    self.logger.warning(
                        "A pending autosave stash is already being tracked; "
                        "preserving the original restore target for this run."
                    )
                elif restore_stash:
                    self._autosave_stash_ref = stash_ref
                    self._autosave_stash_sha = stash_sha
                    self._pending_stash_pop = True

                if stash_ref:
                    self.logger.info(f"Your changes were stashed as {stash_ref}.")
                    self.logger.info(f"Restore with: git stash pop {stash_ref}")
                if stash_sha:
                    self.logger.info(f"Or (stable): git stash apply {stash_sha}")

        self.git.fetch()

        # Ensure base exists locally and is reasonably current.
        self.git.checkout(base)
        self.git.run(["pull", self.git.remote, base], check=False)

        if not self.git.branch_exists(final):
            self.git.checkout(final, create=True, start_point=base)
        else:
            self.git.checkout(final)

        self.logger.info(f"Created/checked out branch {final} from {base}")

        if stashed_this_call:
            if pending_restore:
                self.logger.info(
                    "An earlier autosave stash remains the active restore target for this run."
                )
            elif restore_stash:
                self.logger.info(
                    "Stashed local changes; they will be restored automatically "
                    "when returning to the original branch."
                )
            else:
                self.logger.info(
                    "Stashed local changes; automatic restore is disabled for this branch preparation."
                )

        return final

    def commit_and_push(self, branch: str, title: str) -> None:
        self.client.ensure_user(settings.github_bot_name, settings.github_bot_email)
        self.client.add_all()
        if not self.client.has_changes():
            raise GitError("No changes to commit")
        self.client.commit(title)
        self.client.push(branch)

    def push_branch(self, branch: str) -> None:
        """Push branch without committing (controller already committed)."""
        self.client.push(branch)

    def open_pr(self, branch: str, proposal: Proposal, extra_tests: str = "") -> str:
        if not self._github_ready():
            return ""

        base_branch = settings.github_default_branch
        if not branch or branch == base_branch:
            self.logger.error(
                f"Refusing to open PR with invalid branch selection: head={branch!r} base={base_branch!r}"
            )
            return ""

        title = proposal.title.strip() or "Repo Janitor improvement"
        change_lines = "\n".join(f"- {ch.path} ({ch.apply_mode})" for ch in proposal.changes)
        if not change_lines:
            change_lines = "- Safe automated fixes"

        body = (
            "## Summary\n"
            f"{proposal.description or 'Automated repository improvement.'}\n\n"
            "## Implementation Details\n"
            f"{change_lines}\n\n"
            "## Why This Matters\n"
            f'This change was proposed by Aerith to address: *"{title}"*\n\n'
            "## Tests / Validation\n"
            f"{extra_tests or '- ⚠️ No tests were run'}\n"
        )

        url = f"https://api.github.com/repos/{self.repo_slug}/pulls"
        resp = self._github_request(
            "POST",
            url,
            json={
                "title": title,
                "body": body,
                "head": branch,
                "base": base_branch,
            },
        )
        if resp.status_code not in (200, 201):
            self.logger.error(f"Failed to open PR: {resp.status_code} {resp.text}")
            return ""

        return resp.json().get("html_url", "")

    def update_pr_body(self, branch: str, extra: str) -> None:
        if not self._github_ready():
            return

        try:
            owner = self._repo_owner()
        except ValueError as e:
            self.logger.error(str(e))
            return

        list_url = f"https://api.github.com/repos/{self.repo_slug}/pulls"
        resp = self._github_request(
            "GET",
            list_url,
            params={
                "state": "open",
                "head": f"{owner}:{branch}",
                "base": settings.github_default_branch,
                "per_page": 1,
            },
        )
        if resp.status_code != 200:
            self.logger.error(f"Failed to fetch PRs: {resp.status_code} {resp.text}")
            return

        prs = resp.json()
        if not prs:
            self.logger.error(f"No open PR found for branch {branch}")
            return

        pr = prs[0]
        pr_number = pr["number"]
        new_body = (pr.get("body") or "") + "\n" + extra

        update_url = f"{list_url}/{pr_number}"
        resp2 = self._github_request("PATCH", update_url, json={"body": new_body})
        if resp2.status_code not in (200, 201):
            self.logger.error(f"Failed to update PR body: {resp2.status_code} {resp2.text}")

    def run_tests_and_update_pr(self, branch: str) -> str:
        """
        Run pytest after opening a PR and update its body with results.
        Does not crash if pytest is missing.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                check=False,
                cwd=str(self.root),
                capture_output=True,
                text=True,
            )
            passed = result.returncode == 0
            tail = "\n".join((result.stdout or "").splitlines()[-15:])
            test_report = tail or (result.stderr or "").strip()
        except Exception as e:
            passed = False
            test_report = f"⚠️ Failed to run tests: {e}"

        stamp = "✅ All tests passed" if passed else "❌ Some tests failed"
        body_append = f"\n\n## Test Results\n{stamp}\n```\n{test_report}\n```"
        self.update_pr_body(branch, body_append)
        return test_report

    def restore_original_branch(self) -> None:
        """Switch back to the user's original branch and restore any tracked autosave stash."""
        if not self.original_branch:
            return

        target_branch = self.original_branch
        stash_ref = self._autosave_stash_ref
        stash_sha = self._autosave_stash_sha
        should_restore_stash = bool(self._pending_stash_pop and (stash_ref or stash_sha))

        try:
            self.client.safe_switch(target_branch)
            self.logger.info(f"Restored user branch: {target_branch}")
        except Exception as e:
            self.logger.error(f"Failed to restore branch: {e}")
            return

        if should_restore_stash:
            if stash_ref:
                self.logger.info(f"Your changes were stashed as {stash_ref}.")
                self.logger.info(f"Restore with: git stash pop {stash_ref}")
            if stash_sha:
                self.logger.info(f"Or (stable): git stash apply {stash_sha}")

            if stash_sha:
                self.logger.info(f"Restoring stashed changes (stable SHA): {stash_sha}")
                self.git.stash_apply(stash_sha)

                ref = self.git.find_stash_ref_by_sha(stash_sha)
                if ref:
                    self.logger.info(f"Dropping restored stash entry: {ref}")
                    self.git.stash_drop(ref)
            elif stash_ref:
                self.logger.info(f"Restoring stashed changes (explicit ref): {stash_ref}")
                self.git.stash_pop(stash_ref)

        self.original_branch = None
        self._autosave_stash_ref = None
        self._autosave_stash_sha = None
        self._pending_stash_pop = False
