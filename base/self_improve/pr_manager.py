
from __future__ import annotations
from typing import Optional
from loguru import logger
from pathlib import Path
import os
import requests

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

    def open_pr(self, branch: str, title: str, body: str) -> str:
        if not self.repo_slug or not self.token:
            raise RuntimeError("GitHub credentials missing. Set GITHUB_TOKEN and GITHUB_REPO.")
        url = f"https://api.github.com/repos/{self.repo_slug}/pulls"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }
        payload = {
            "title": title,
            "head": branch,
            "base": settings.github_default_branch,
            "body": body,
            "maintainer_can_modify": True,
            "draft": False,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code >= 300:
            logger.error(f"GitHub PR create failed: {r.status_code} {r.text}")
            raise RuntimeError(f"GitHub PR create failed: {r.text}")
        pr_url = r.json().get("html_url", "")
        return pr_url

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
