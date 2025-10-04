from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field


def _read_secret(path_env: str) -> str | None:
    path = os.getenv(path_env)
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8").strip()
    return None


github_ssh_key = _read_secret("GITHUB_SSH_KEY_FILE")
github_gpg_key = _read_secret("GITHUB_GPG_KEY_FILE")


env_override = os.getenv("ULTRON_ENV_PATH")
if env_override and os.path.exists(env_override):
    load_dotenv(env_override, override=True)
else:
    # 2) Otherwise, find the nearest .env (project root)
    found = find_dotenv(filename=".env", usecwd=True)
    if found:
        load_dotenv(found, override=True)
    else:
        load_dotenv(override=True)  # last-resort fallback


class Settings(BaseModel):
    # simple, concrete types (no Optional for mode)
    mode: Literal["text", "voice", "stream"] = "text"

    db_path: str = Field(default=os.getenv("ULTRON_DB_PATH", "./ultron.db"))
    memory_ttl_days: int = int(os.getenv("ULTRON_MEMORY_TTL_DAYS", 30))
    importance_threshold: int = int(os.getenv("ULTRON_IMPORTANCE_THRESHOLD", 25))

    # LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

    # ElevenLabs
    eleven_api_key: str | None = os.getenv("ELEVENLABS_API_KEY")
    eleven_voice_id: str | None = os.getenv("ELEVENLABS_VOICE_ID")

    # Home Assistant
    ha_base_url: str | None = os.getenv("HA_BASE_URL")
    ha_token: str | None = os.getenv("HA_TOKEN")

    # Consolidation cron
    consolidation_hour: int = int(os.getenv("ULTRON_CONSOLIDATION_HOUR", 3))
    consolidation_minute: int = int(os.getenv("ULTRON_CONSOLIDATION_MINUTE", 0))

    # Voice
    auto_speak: bool = False  # ✅ default enabled
    wake_word: str = "ultron"  # ✅ wake word for vocal cues
    wake_commands: dict[str, str] = {
        "text me": "disable_speak",  # Ultron will text instead of speak
        "talk to me": "enable_speak",  # Explicitly turn speaking back on
    }

    # --- GitHub / PR settings ---
    github_token: str | None = os.getenv("GITHUB_TOKEN")  # required for PRs
    github_repo: str | None = os.getenv("GITHUB_REPO")  # e.g. "yourname/assistant"
    github_default_branch: str = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    github_bot_name: str = os.getenv("GITHUB_BOT_NAME", "ultron-bot")
    github_bot_email: str = os.getenv("GITHUB_BOT_EMAIL", "ultron-bot@local")
    github_remote_name: str = os.getenv("GITHUB_REMOTE_NAME", "origin")

    # Proposer behavior
    proposer_allowlist: list[str] = [
        "base/",
        "config/",
        "run.py",
    ]
    proposer_blocklist: list[str] = [".venv/", ".git/", "data/", "models/", "__pycache__/"]
    proposer_branch_prefix: str = os.getenv("PROPOSER_BRANCH_PREFIX", "ultron/proposal/")
    proposer_max_files_per_pr: int = int(os.getenv("PROPOSER_MAX_FILES_PER_PR", "20"))
    proposer_max_patch_bytes: int = int(os.getenv("PROPOSER_MAX_PATCH_BYTES", str(256_000)))
    proposal_notify_stdout: bool = True  # simple notification channel; you can add Slack later


settings = Settings()

# class Settings(BaseModel):
#     mode: str = os.getenv("ULTRON_MODE", "text")

# settings = Settings()
