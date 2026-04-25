from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_binary(repo_root: Path, *names: str) -> str | None:
    search_dirs = [
        repo_root / "node_modules" / ".bin",
        repo_root / ".venv" / "bin",
    ]

    if os.name == "nt":
        suffixes = (".cmd", ".exe", ".bat", "")
    else:
        suffixes = ("",)

    for directory in search_dirs:
        for name in names:
            for suffix in suffixes:
                candidate = directory / f"{name}{suffix}"
                if candidate.exists() and candidate.is_file():
                    return str(candidate)

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    return None
