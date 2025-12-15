"""Lightweight stub of python-dotenv for offline environments."""
from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(filename: str = ".env", usecwd: bool = False) -> str:
    """Return the path to a .env file if one exists."""
    start_dir = Path.cwd() if usecwd else Path(__file__).resolve().parent
    for parent in [start_dir, *start_dir.parents]:
        candidate = parent / filename
        if candidate.exists():
            return str(candidate)
    return ""


def load_dotenv(path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
    """Minimal loader that sets environment variables from a .env-style file."""
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if override or key not in os.environ:
            os.environ[key] = value.strip()
    return True

