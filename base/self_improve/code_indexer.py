
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import ast
from pathlib import Path
from loguru import logger

@dataclass
class CodeSymbol:
    kind: str      # "class" | "def"
    name: str
    lineno: int
    col: int

@dataclass
class FileIndex:
    path: str
    symbols: List[CodeSymbol]

class CodeIndexer:
    def __init__(self, root: str, allowlist: List[str], blocklist: List[str]):
        self.root = Path(root).resolve()
        self.allowlist = [str((self.root / a).resolve()) for a in allowlist]
        self.blocklist = [str((self.root / b).resolve()) for b in blocklist]

    def _allowed(self, path: Path) -> bool:
        sp = str(path.resolve())
        if any(sp.startswith(b) for b in self.blocklist):
            return False
        return any(sp.startswith(a) for a in self.allowlist)

    def scan(self) -> List[FileIndex]:
        files: List[FileIndex] = []
        for p in self.root.rglob("*.py"):
            if not self._allowed(p):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(txt)
                symbols: List[CodeSymbol] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        symbols.append(CodeSymbol("def", node.name, node.lineno, node.col_offset))
                    elif isinstance(node, ast.ClassDef):
                        symbols.append(CodeSymbol("class", node.name, node.lineno, node.col_offset))
                files.append(FileIndex(str(p.relative_to(self.root)), symbols))
            except Exception as e:
                logger.debug(f"Index skip {p}: {e}")
        return files

    @staticmethod
    def to_markdown(index: List[FileIndex]) -> str:
        lines = []
        for fi in index:
            lines.append(f"- `{fi.path}`")
            for s in fi.symbols[:50]:
                lines.append(f"  - {s.kind} {s.name} @ L{s.lineno}")
        return "\n".join(lines)
