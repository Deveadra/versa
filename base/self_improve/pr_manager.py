# base/self_improve/pr_manager.py
from __future__ import annotations
from typing import Optional
from loguru import logger
from pathlib import Path
import os, sys, subprocess, requests

from base.self_improve.models import Proposal
from base.devops.git_client import GitClient, GitError
from config.config import settings

class PRManager:
    """
    Creates a branch, commits applied changes, pushes, opens a PR via GitHub API.
    """
    def __init__(self, repo_root: str, repo_slug: Optional[str] = None, token: Optional[str] = None):
        self.root = Path(repo_root).resolve()
        self.repo_slug = repo_slug or settings.github_repo
        self.token = token or settings.github_token
        self.client = GitClient(self.root, remote=settings.github_remote_name)
        self.original_branch: Optional[str] = None

    def prepare_branch(self, name_suffix: str) -> str:
        """
        Prepare or reuse a proposal branch.
        - Remember original branch
        - Fetch + sync default branch
        - Create new branch off default (or reuse if exists)
        - Return branch name
        """
        base = settings.github_default_branch
        branch = f"{settings.proposer_branch_prefix}{name_suffix}"

        # Remember where the user was
        try:
            self.original_branch = self.client.current_branch()
        except Exception:
            self.original_branch = base

        # If already on desired branch, reuse it
        if self.original_branch == branch:
            logger.info(f"Already on {branch}, reusing it")
            return branch

        # Sync default branch
        self.client.fetch()
        self.client.safe_switch(base)

        # Try to create branch from base; otherwise reuse
        try:
            self.client.checkout(branch, create=True, start_point=base)
            logger.info(f"Created new branch {branch} from {base}")
        except GitError as e:
            if "already exists" in str(e):
                logger.info(f"Reusing existing branch {branch}")
                self.client.checkout(branch)
            else:
                raise

        return branch

    def commit_and_push(self, branch: str, title: str) -> None:
        self.client.ensure_user(settings.github_bot_name, settings.github_bot_email)
        self.client.add_all()
        if not self.client.has_changes():
            raise GitError("No changes to commit")
        self.client.commit(title)
        self.client.push(branch)

    def open_pr(self, branch: str, proposal: Proposal, extra_tests: str = "") -> str:
        title = proposal.title
        body = f"""## Summary
{proposal.description}

## Implementation Details
{chr(10).join(f"- {ch.path} ({ch.apply_mode})" for ch in proposal.changes)}

## Why This Matters
This change was proposed by Ultron to address: *"{title}"*

## Tests / Validation
{extra_tests or "- ⚠️ No tests were run"}
"""

        url = f"https://api.github.com/repos/{settings.github_repo}/pulls"
        headers = {"Authorization": f"token {settings.github_token}"}
        resp = requests.post(
            url,
            headers=headers,
            json={
                "title": title,
                "body": body,
                "head": branch,
                "base": settings.github_default_branch,
            },
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Failed to open PR: {resp.text}")
            return ""
        return resp.json().get("html_url", "")

    def update_pr_body(self, branch: str, extra: str) -> None:
        url = f"https://api.github.com/repos/{settings.github_repo}/pulls"
        headers = {"Authorization": f"token {settings.github_token}"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch PRs: {resp.text}")
            return
        prs = resp.json()
        pr = next((p for p in prs if p["head"]["ref"] == branch), None)
        if not pr:
            logger.error(f"No PR found for branch {branch}")
            return

        pr_number = pr["number"]
        new_body = (pr.get("body") or "") + "\n" + extra
        update_url = f"{url}/{pr_number}"
        resp2 = requests.patch(update_url, headers=headers, json={"body": new_body})
        if resp2.status_code not in (200, 201):
            logger.error(f"Failed to update PR body: {resp2.text}")

    def run_tests_and_update_pr(self, branch: str) -> str:
        """
        Run pytest after opening a PR and update its body with results.
        Does not crash if pytest is missing.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(self.root),
                capture_output=True,
                text=True
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

    def restore_original_branch(self):
        """Switch back to the branch the user was on."""
        if self.original_branch:
            try:
                self.client.safe_switch(self.original_branch)
                logger.info(f"Restored user branch: {self.original_branch}")
            except Exception as e:
                logger.error(f"Failed to restore branch: {e}")
