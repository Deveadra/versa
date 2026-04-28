from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path


def resolve_binary(repo_root: Path, *names: str) -> str | None:
    search_dirs = [
        repo_root / "node_modules" / ".bin",
        repo_root / ".venv" / "bin",
    ]

    suffixes = (".cmd", ".exe", ".bat", "") if os.name == "nt" else ("",)

    for directory in search_dirs:
        for name, suffix in itertools.product(names, suffixes):
            candidate = directory / f"{name}{suffix}"
            if candidate.exists() and candidate.is_file():
                return str(candidate)

    for name in names:
        if resolved := shutil.which(name):
            return resolved

    return None
