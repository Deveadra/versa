
from __future__ import annotations
import re, os, ast
import time, tracemalloc
from pathlib import Path
from loguru import logger

class DiagnosticEngine:
    def __init__(self, repo_root: str):
        self.root = Path(repo_root).resolve()

    def scan(self):
        """
        Return (conversational_report, structured_json)
        structured_json = { "issues": [ { "file":..., "summary":... } ], "fixable": True/False }
        """
        issues = []

        # Example: simple static scans
        for pyfile in self.root.rglob("*.py"):
            try:
                text = pyfile.read_text(encoding="utf-8", errors="ignore")
                # Quick syntax check
                try:
                    ast.parse(text)
                except SyntaxError as e:
                    issues.append({"file": str(pyfile), "summary": f"Syntax error at line {e.lineno}"})
                # Style scan: trailing whitespace
                if re.search(r"[ \t]+$", text, re.M):
                    issues.append({"file": str(pyfile), "summary": "Trailing whitespace"})
            except Exception as e:
                logger.debug(f"Scan failed for {pyfile}: {e}")

        structured = {"issues": issues, "fixable": bool(issues)}
        return "Scan complete", structured

    def benchmark_action(label: str, func, *args, **kwargs) -> dict:
      """Run func, measure latency and memory, return stats + output."""
      start = time.perf_counter()
      tracemalloc.start()
      try:
          result = func(*args, **kwargs)
      finally:
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()
      end = time.perf_counter()
      return {
          "label": label,
          "latency_ms": round((end - start) * 1000, 2),
          "mem_kb": round(peak / 1024, 2),
          "result": str(result)[:200]  # preview
      }
      
    def apply_fixes(self, structured):
        """
        Try to auto-fix simple issues like whitespace.
        Returns list of applied fixes.
        """
        applied = []
        for issue in structured.get("issues", []):
            if "Trailing whitespace" in issue["summary"]:
                p = Path(issue["file"])
                text = p.read_text(encoding="utf-8", errors="ignore")
                fixed = re.sub(r"[ \t]+$", "", text, flags=re.M)
                p.write_text(fixed, encoding="utf-8")
                applied.append(issue)
        return applied
