# config/self_improvement.py
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class SelfImproveConfig:
    # Paths
    repo_root: str = os.getenv("ULTRON_REPO_ROOT", ".")
    logs_dir: str = os.getenv("ULTRON_LOGS_DIR", "logs")
    learning_dir: str = os.getenv("ULTRON_LEARNING_DIR", "memory/learning")
    reports_dir: str = os.getenv("ULTRON_REPORTS_DIR", "memory/reports")

    # Git / MR
    git_remote: str = os.getenv("ULTRON_GIT_REMOTE", "origin")
    git_default_branch: str = os.getenv("ULTRON_GIT_DEFAULT_BRANCH", "main")
    gitlab_api_url: str = os.getenv("ULTRON_GITLAB_API_URL", "").rstrip("/")
    gitlab_project_id: str = os.getenv("ULTRON_GITLAB_PROJECT_ID", "")
    gitlab_token: str = os.getenv("ULTRON_GITLAB_TOKEN", "")

    # Diagnostics
    ruff_args: str = os.getenv("ULTRON_RUFF_ARGS", "ruff --config pyproject.toml check .")
    mypy_args: str = os.getenv("ULTRON_MYPY_ARGS", "mypy --install-types --non-interactive")
    pytest_collect_args: str = os.getenv("ULTRON_PYTEST_COLLECT_ARGS", "pytest -q --collect-only")
    perf_sample_seconds: int = int(os.getenv("ULTRON_PERF_SAMPLE_SECONDS", "5"))

    # Proposal thresholds
    open_mr_on_any_fix: bool = os.getenv("ULTRON_OPEN_MR_ON_ANY_FIX", "1") == "1"

CFG = SelfImproveConfig()
