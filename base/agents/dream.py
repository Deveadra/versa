# base/agents/dream.py
from __future__ import annotations
import os, json, time, glob, traceback
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from config.self_improvements import CFG

class DreamCycle:
    """
    Aggregates the day's signals (logs, corrections, failures), produces insights,
    and stores them for the diagnostics/proposal engine.
    """
    def __init__(self, now: datetime | None = None):
        self.now = now or datetime.utcnow()
        Path(CFG.learning_dir).mkdir(parents=True, exist_ok=True)

    def _collect_log_snippets(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        if not os.path.isdir(CFG.logs_dir):
            return result
        for path in glob.glob(os.path.join(CFG.logs_dir, "**/*.*"), recursive=True):
            try:
                # keep tail of each file to avoid huge payloads
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-400:]
                key = os.path.relpath(path, CFG.logs_dir)
                result[key] = [ln.rstrip("\n") for ln in lines]
            except Exception:
                # best-effort
                continue
        return result

    def _summarize_patterns(self, logs: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Lightweight heuristic summaries (no LLM dependency here).
        """
        summary: Dict[str, Any] = {"errors": {}, "warnings": {}, "notes": []}
        for file, lines in logs.items():
            for ln in lines:
                low = ln.lower()
                if "traceback" in low or "error" in low or "exception" in low:
                    summary["errors"].setdefault(file, 0)
                    summary["errors"][file] += 1
                elif "warn" in low or "deprec" in low:
                    summary["warnings"].setdefault(file, 0)
                    summary["warnings"][file] += 1
        # naive hypotheses
        if summary["errors"]:
            summary["notes"].append("Recurring errors found; prioritize diagnostics & tests near top offenders.")
        if summary["warnings"]:
            summary["notes"].append("Warnings present; consider dependency pins and small refactors.")
        if not summary["errors"] and not summary["warnings"]:
            summary["notes"].append("Day appears clean; focus on performance and reliability improvements.")
        return summary

    def run(self) -> str:
        t0 = time.time()
        logs = self._collect_log_snippets()
        summary = self._summarize_patterns(logs)
        payload = {
            "timestamp": self.now.isoformat(),
            "logs_considered": list(logs.keys()),
            "summary": summary,
            "duration_sec": round(time.time() - t0, 3),
        }
        out_path = os.path.join(
            CFG.learning_dir,
            f"dream_summary_{self.now.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path
