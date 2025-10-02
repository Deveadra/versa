
from __future__ import annotations
from typing import Optional
from loguru import logger
from pathlib import Path
import os
import requests
import subprocess

from base.self_improve.proposal_engine import Proposal
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

    def commit_and_push(self, branch: str, title: str) -> None:
        self.client.ensure_user(settings.github_bot_name, settings.github_bot_email)
        self.client.add_all()
        if not self.client.has_changes():
            raise GitError("No changes to commit")
        self.client.commit(title)
        self.client.push(branch)

    def prepare_branch(self, name_suffix: str) -> str:
        branch = settings.proposer_branch_prefix + name_suffix
        self.client.fetch()
        # Start from default branch
        self.client.checkout(settings.github_default_branch)
        # Create from remote default baseline if exists
        self.client.checkout(branch, create=True)
        return branch
    
    def update_pr_body(self, branch: str, extra: str) -> None:
        """
        Append extra info (like test results) to the PR body.
        """
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
        new_body = pr["body"] + "\n" + extra
        update_url = f"{url}/{pr_number}"
        resp2 = requests.patch(
            update_url,
            headers=headers,
            json={"body": new_body},
        )
        if resp2.status_code not in (200, 201):
            logger.error(f"Failed to update PR body: {resp2.text}")
            
    def run_tests_and_update_pr(self, branch: str) -> str:
        """
        Run pytest after opening a PR and update its body with results.
        """

        try:
            result = subprocess.run(
                ["pytest", "--maxfail=5", "--disable-warnings", "-q"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=120
            )
            passed = result.returncode == 0
            summary = result.stdout.strip().splitlines()[-10:]
            test_report = "\n".join(summary)
        except Exception as e:
            passed = False
            test_report = f"⚠️ Failed to run tests: {e}"

        body_append = f"\n\n## Test Results\n{'✅ All tests passed' if passed else '❌ Some tests failed'}\n```\n{test_report}\n```"

        self.update_pr_body(branch, body_append)
        return test_report
