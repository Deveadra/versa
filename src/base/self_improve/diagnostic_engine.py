from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
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
                    issues.append(
                        {"file": str(pyfile), "summary": f"Syntax error at line {e.lineno}"}
                    )
                # Style scan: trailing whitespace
                if re.search(r"[ \t]+$", text, re.M):
                    issues.append({"file": str(pyfile), "summary": "Trailing whitespace"})
            except Exception as e:
                logger.debug(f"Scan failed for {pyfile}: {e}")

        structured = {"issues": issues, "fixable": bool(issues)}
        return "Scan complete", structured

    def benchmark_action(self, label: str, func, *args, **kwargs) -> dict:
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
            "result": str(result)[:200],  # preview
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

    def _run_tool(self, cmd: list[str], name: str) -> dict:
        """
        Run a diagnostic tool safely, capturing 'not installed' and other failures.
        Returns: {tool, exit_code, stdout, stderr, duration_sec}
        """
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.root,
            )
            out, err = proc.communicate()
            rc = proc.returncode
        except FileNotFoundError as e:
            out, err, rc = "", str(e), 127
        except Exception as e:
            out, err, rc = "", f"{type(e).__name__}: {e}", 99

        return {
            "tool": name,
            "exit_code": rc,
            "stdout": (out or "")[-200_000:],
            "stderr": (err or "")[-200_000:],
            "duration_sec": round(time.time() - t0, 3),
        }

    def run(self) -> str:
        """
        Orchestrate diagnostics and write JSON report to memory/reports/.
        Returns the path to the report file.
        """
        started_at_iso = datetime.utcnow().isoformat()

        results = []
        results.append(self._run_tool([sys.executable, "-m", "black", "--check", "."], "black"))
        results.append(self._run_tool([sys.executable, "-m", "ruff", "check", "."], "ruff"))
        results.append(
            self._run_tool([sys.executable, "-m", "pytest", "-q", "--collect-only"], "pytest")
        )

        any_fail = any(r["exit_code"] != 0 for r in results)
        report = {
            "started_at": started_at_iso,
            "finished_at": datetime.utcnow().isoformat(),
            "tool_results": results,
            "any_failures": any_fail,
        }

        out_dir = self.root / "memory" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"diagnostic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        from loguru import logger

        logger.info(f"[diagnostics] wrote report → {out_path}")
        return str(out_path)
