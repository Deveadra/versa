from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


@dataclass
class CodeSymbol:
    kind: str  # "class" | "def" | "async def"
    name: str
    lineno: int
    col: int


class CodeIndexer:
    """
    Robust repo indexer used for proposal context.

    - root MUST be the repo root (Orchestrator passes Path(".").resolve()).
    - allowlist/blocklist are glob patterns (e.g. "src/base/**").
    - scan() returns a dict suitable for persistence + LLM context.
    - to_markdown() accepts that dict (no type mismatch).
    """

    def __init__(self, root: str, allowlist: list[str], blocklist: list[str]):
        self.root = Path(root).resolve()  # repo root
        self.allowlist = tuple(a.replace("\\", "/") for a in (allowlist or ["**/*.py"]))
        self.blocklist = tuple(b.replace("\\", "/") for b in (blocklist or []))

    def _cache_path(self) -> Path:
        d = self.root / ".aerith"
        d.mkdir(parents=True, exist_ok=True)
        return d / "index_cache.json"

    def _load_cache(self) -> dict[str, Any]:
        p = self._cache_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        p = self._cache_path()
        try:
            p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def _allowed(self, relpath: str) -> bool:
        rp = relpath.replace("\\", "/")
        if any(fnmatch(rp, pat) for pat in self.blocklist):
            return False
        return any(fnmatch(rp, pat) for pat in self.allowlist)

    def _index_file(self, p: Path) -> dict[str, Any]:
        relpath = self._rel(p)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {"path": relpath, "summary": "", "symbols": []}

        summary = ""
        try:
            tree = ast.parse(txt)
            doc = ast.get_docstring(tree) or ""
            summary = doc.strip().splitlines()[0] if doc.strip() else ""
            symbols: list[dict[str, Any]] = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    symbols.append(
                        {"kind": "class", "name": node.name, "lineno": node.lineno, "col": node.col_offset}
                    )
                elif isinstance(node, ast.FunctionDef):
                    symbols.append(
                        {"kind": "def", "name": node.name, "lineno": node.lineno, "col": node.col_offset}
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.append(
                        {"kind": "async def", "name": node.name, "lineno": node.lineno, "col": node.col_offset}
                    )
        except Exception:
            symbols = []
        return {"path": relpath, "summary": summary, "symbols": symbols}

    def scan(self, incremental: bool = True) -> dict[str, Any]:
        """
        Returns:
          {
            "files": {
              "src/base/...py": {"mtime": float, "summary": str, "symbols": [...]},
              ...
            }
          }
        """
        cache = self._load_cache() if incremental else {}
        files_cache = cache.get("files", {}) if incremental else {}
        out: dict[str, Any] = {"files": {}}

        for p in self.root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue

            rp = self._rel(p)
            if not self._allowed(rp):
                continue

            try:
                mtime = p.stat().st_mtime
            except Exception:
                continue

            if incremental and rp in files_cache and files_cache[rp].get("mtime") == mtime:
                out["files"][rp] = files_cache[rp]
                continue

            info = self._index_file(p)
            info["mtime"] = mtime
            out["files"][rp] = info

        if incremental:
            self._save_cache(out)
        return out

    @staticmethod
    def to_markdown(index: dict[str, Any]) -> str:
        files = (index or {}).get("files", {}) or {}
        lines: list[str] = ["# Repository Index", ""]
        for path in sorted(files.keys()):
            info = files[path] or {}
            summary = info.get("summary") or ""
            lines.append(f"- `{path}`" + (f" — {summary}" if summary else ""))
            syms = info.get("symbols") or []
            for s in syms[:12]:
                lines.append(f"  - {s.get('kind')} {s.get('name')} @ L{s.get('lineno')}")
        return "\n".join(lines)