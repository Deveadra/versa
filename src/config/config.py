from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field


def _read_secret(path_env: str) -> str | None:
    """
    Read a secret value from a file path stored in an environment variable.
    Example:
        GITHUB_SSH_KEY_FILE=/path/to/key
    """
    path = os.getenv(path_env)
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8").strip()
    return None


def _get_int(env: str, default: int) -> int:
    raw = os.getenv(env)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(env: str, default: bool = False) -> bool:
    raw = os.getenv(env)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# ------------------------------------------------------------
# Load environment FIRST (so anything below sees the .env values)
# ------------------------------------------------------------
env_override = os.getenv("AERITH_ENV_PATH")
if env_override and os.path.exists(env_override):
    load_dotenv(env_override, override=True)
else:
    found = find_dotenv(filename=".env", usecwd=True)
    if found:
        load_dotenv(found, override=True)
    else:
        load_dotenv(override=True)  # last-resort fallback


class Settings(BaseModel):
    # Runtime mode (allow env override)
    mode: Literal["text", "voice", "stream"] = Field(
        default_factory=lambda: os.getenv("AERITH_MODE", "text")  # type: ignore[return-value]
    )

    # Core storage
    db_path: str = Field(default_factory=lambda: os.getenv("AERITH_DB_PATH", "./aerith.db"))
    memory_ttl_days: int = Field(default_factory=lambda: _get_int("AERITH_MEMORY_TTL_DAYS", 30))
    importance_threshold: int = Field(
        default_factory=lambda: _get_int("AERITH_IMPORTANCE_THRESHOLD", 25)
    )

    # LLM
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # TTS engine choice (single source of truth)
    # valid examples: "elevenlabs", "aerith"
    tts_engine: str = Field(default_factory=lambda: os.getenv("TTS_ENGINE", "elevenlabs"))

    # Embeddings
    embeddings_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers")
    )
    embeddings_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    )

    # ElevenLabs
    eleven_api_key: str | None = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY"))
    eleven_voice_id: str | None = Field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID"))

    # Home Assistant
    ha_base_url: str | None = Field(default_factory=lambda: os.getenv("HA_BASE_URL"))
    ha_token: str | None = Field(default_factory=lambda: os.getenv("HA_TOKEN"))

    # Consolidation cron
    consolidation_hour: int = Field(default_factory=lambda: _get_int("AERITH_CONSOLIDATION_HOUR", 3))
    consolidation_minute: int = Field(
        default_factory=lambda: _get_int("AERITH_CONSOLIDATION_MINUTE", 0)
    )

    # Voice UX
    auto_speak: bool = Field(default_factory=lambda: _get_bool("AERITH_AUTO_SPEAK", False))
    wake_word: str = Field(default_factory=lambda: os.getenv("AERITH_WAKE_WORD", "aerith"))
    wake_commands: dict[str, str] = {
        "text me": "disable_speak",
        "talk to me": "enable_speak",
    }

    # --- GitHub / PR settings ---
    github_token: str | None = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    github_repo: str | None = Field(default_factory=lambda: os.getenv("GITHUB_REPO"))
    github_default_branch: str = Field(default_factory=lambda: os.getenv("GITHUB_DEFAULT_BRANCH", "main"))
    github_bot_name: str = Field(default_factory=lambda: os.getenv("GITHUB_BOT_NAME", "aerith-bot"))
    github_bot_email: str = Field(default_factory=lambda: os.getenv("GITHUB_BOT_EMAIL", "aerith-bot@local"))
    github_remote_name: str = Field(default_factory=lambda: os.getenv("GITHUB_REMOTE_NAME", "origin"))

    # Optional: key material from files (handy for git signing / SSH operations)
    github_ssh_key: str | None = Field(default_factory=lambda: _read_secret("GITHUB_SSH_KEY_FILE"))
    github_gpg_key: str | None = Field(default_factory=lambda: _read_secret("GITHUB_GPG_KEY_FILE"))

    # Proposer behavior
    proposer_allowlist: list[str] = [
        "src/base/",
        "src/config/",
        "scripts/",
        "tests/",
        ".github/workflows/",
        "pyproject.toml",
        "README.md",
        "run.py",
    ]
    proposer_blocklist: list[str] = [
        ".venv/",
        ".git/",
        "data/",
        "models/",
        "__pycache__/",
    ]
    proposer_branch_prefix: str = Field(
        default_factory=lambda: os.getenv("PROPOSER_BRANCH_PREFIX", "aerith/proposal/")
    )
    proposer_max_files_per_pr: int = Field(
        default_factory=lambda: _get_int("PROPOSER_MAX_FILES_PER_PR", 20)
    )
    proposer_max_patch_bytes: int = Field(
        default_factory=lambda: _get_int("PROPOSER_MAX_PATCH_BYTES", 256_000)
    )
    proposal_notify_stdout: bool = True

    # Qdrant vector database settings
    qdrant_url: str | None = Field(default_factory=lambda: os.getenv("QDRANT_URL"))
    qdrant_api_key: str | None = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "events"))

    # ------------------------------------------------------------
    # Self-improvement / dream-cycle settings (single source of truth)
    # ------------------------------------------------------------
    self_improve_enabled: bool = Field(
        default_factory=lambda: _get_bool("AERITH_SELF_IMPROVE_ENABLED", False)
    )

    repo_root: str = Field(default_factory=lambda: os.getenv("AERITH_REPO_ROOT", "."))
    logs_dir: str = Field(default_factory=lambda: os.getenv("AERITH_LOGS_DIR", "logs"))
    learning_dir: str = Field(
        default_factory=lambda: os.getenv("AERITH_LEARNING_DIR", "memory/learning")
    )
    reports_dir: str = Field(
        default_factory=lambda: os.getenv("AERITH_REPORTS_DIR", "memory/reports")
    )

    git_remote: str = Field(default_factory=lambda: os.getenv("AERITH_GIT_REMOTE", "origin"))
    git_default_branch: str = Field(
        default_factory=lambda: os.getenv("AERITH_GIT_DEFAULT_BRANCH", "main")
    )

    ruff_args: str = Field(
        default_factory=lambda: os.getenv(
            "AERITH_RUFF_ARGS", "ruff --config pyproject.toml check ."
        )
    )
    mypy_args: str = Field(
        default_factory=lambda: os.getenv(
            "AERITH_MYPY_ARGS", "mypy --install-types --non-interactive"
        )
    )
    pytest_collect_args: str = Field(
        default_factory=lambda: os.getenv(
            "AERITH_PYTEST_COLLECT_ARGS", "pytest -q --collect-only"
        )
    )
    perf_sample_seconds: int = Field(
        default_factory=lambda: _get_int("AERITH_PERF_SAMPLE_SECONDS", 5)
    )
    open_mr_on_any_fix: bool = Field(
        default_factory=lambda: _get_bool("AERITH_OPEN_MR_ON_ANY_FIX", True)
    )
    
    
    @property
    def use_llm(self) -> bool:
        # computed AFTER dotenv load
        return bool(self.openai_api_key)


settings = Settings()