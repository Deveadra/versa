# base/self_improve/code_indexer.py
from __future__ import annotations

import ast
import json, os, time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from base.self_improve.code_indexer import CodeIndexer


@dataclass
class CodeSymbol:
    kind: str  # "class" | "def"
    name: str
    lineno: int
    col: int


@dataclass
class FileIndex:
    path: str
    symbols: list[CodeSymbol]


class CodeIndexer:
    def __init__(self, root: str, allowlist: list[str], blocklist: list[str]):
        self.root = Path(root).resolve()
        self.repo_root = self.root.parent
        self.allowlist = [str((self.root / a).resolve()) for a in allowlist]
        self.blocklist = [str((self.root / b).resolve()) for b in blocklist]

    def _cache_path(self) -> Path:
        root = Path(self.repo_root)
        return root / ".ultron_index_cache.json"

    def _load_cache(self) -> dict:
        p = self._cache_path()
        if p.exists():
            try:
                return json.load(p.open("r", encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        p = self._cache_path()
        try:
            json.dump(cache, p.open("w", encoding="utf-8"))
        except Exception:
            pass

    # def scan(self) -> list[FileIndex]:
    #     files: list[FileIndex] = []
    #     for p in self.root.rglob("*.py"):
    #         if not self._allowed(p):
    #             continue
    #         try:
    #             txt = p.read_text(encoding="utf-8", errors="ignore")
    #             tree = ast.parse(txt)
    #             symbols: list[CodeSymbol] = []
    #             for node in ast.walk(tree):
    #                 if isinstance(node, ast.FunctionDef):
    #                     symbols.append(CodeSymbol("def", node.name, node.lineno, node.col_offset))
    #                 elif isinstance(node, ast.ClassDef):
    #                     symbols.append(CodeSymbol("class", node.name, node.lineno, node.col_offset))
    #             files.append(FileIndex(str(p.relative_to(self.root)), symbols))
    #         except Exception as e:
    #             logger.debug(f"Index skip {p}: {e}")
    #     return files
    
    def scan(self, incremental: bool = True) -> dict:
        """
        Return a dict index. If incremental, only re-read files whose mtime changed since last scan.
        Cache format: { "files": { relpath: { "mtime": float, "summary": "...", ... } } }
        """
        root = Path(self.repo_root)
        cache = self._load_cache() if incremental else {}
        files_cache = cache.get("files", {}) if incremental else {}
        out = {"files": {}}

        def rel(p: Path) -> str:
            return str(p.relative_to(root)).replace("\\", "/")

        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            rp = rel(p)
            try:
                mtime = p.stat().st_mtime
            except Exception:
                continue

            if incremental and rp in files_cache and files_cache[rp].get("mtime") == mtime:
                out["files"][rp] = files_cache[rp]
                continue

            # re-index this file (whatever you already do: summary, symbols, etc.)
            info = self._index_file(p)  # your existing method
            info["mtime"] = mtime
            out["files"][rp] = info

        # optionally: drop entries that no longer exist
        if incremental:
            existing = set(out["files"].keys())
            for rp in list(files_cache.keys()):
                if rp not in existing:
                    # removed file; skip keeping stale entry
                    pass

        if incremental:
            self._save_cache(out)
        return out
    
    def _allowed(self, path: Path) -> bool:
        sp = str(path.resolve())
        if any(sp.startswith(b) for b in self.blocklist):
            return False
        return any(sp.startswith(a) for a in self.allowlist)
    

    @staticmethod
    def to_markdown(index: list[FileIndex]) -> str:
        lines = []
        for fi in index:
            lines.append(f"- `{fi.path}`")
            for s in fi.symbols[:50]:
                lines.append(f"  - {s.kind} {s.name} @ L{s.lineno}")
        return "\n".join(lines)
