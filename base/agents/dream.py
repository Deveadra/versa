# base/agents/dream.py
from __future__ import annotations
import os, json, time, glob, traceback

from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from base.memory.store import MemoryStore
from base.database.sqlite import SQLiteConn
from base.llm.brain import ask_brain
from config.config import settings
from config.self_improvements import CFG


# @dataclass
# class Event:
#     ts: str
#     role: str          # "user" | "ultron" | "system"
#     text: str
#     meta: Dict[str, Any]
    
class DreamCycle:
    """
    Aggregates the day's signals (logs, corrections, failures), produces a structured summary of insights,
    and stores them (JSON file + memory event) for the diagnostics/proposal engine, and self-reflection.
    """
    def __init__(self, now: Optional[datetime] = None, store: Optional[MemoryStore] = None):
        self.now = now or datetime.utcnow()
        # Ensure summaries directory exists
        Path("data/summaries").mkdir(parents=True, exist_ok=True)
        # Use provided MemoryStore or initialize a new one if needed
        if store is not None:
            self.store = store
        else:
            db = SQLiteConn(settings.db_path)
            self.store = MemoryStore(db.conn)

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

    def run(self) -> Dict[str, Any]:
        """Collect last 24h of events and generate a summary dict."""
        since_ts = (self.now - timedelta(days=1)).isoformat()
        # Fetch events from the past 24 hours
        cur = self.store.conn.cursor()
        cur.execute(
            "SELECT ts, content, type FROM events WHERE ts >= ? ORDER BY ts",
            (since_ts,)
        )
        rows = cur.fetchall()
        if not rows:
            logger.info("DreamCycle: No interactions in the last 24h to summarize.")
            return {"timestamp": self.now.isoformat(), "summary": "", "highlights": [], "learnings": [], "todos": []}
        # Prepare a transcript of interactions for the prompt
        interactions: List[str] = []
        for r in rows:
            # We attempt to distinguish roles if possible
            text = r["content"]
            # Optionally prepend role if type indicates it (e.g., user vs assistant), if such convention is used
            # For simplicity, just use the content as-is:
            interactions.append(text)
        transcript = "\n".join(interactions)
        # Compose prompt for the LLM to summarize the day's interactions
        user_prompt = (
            "Summarize the following interactions between the user and assistant from the last 24 hours.\n"
            "Provide a JSON with keys: summary (overall recap), highlights (notable moments), "
            "learnings (new insights or lessons), and todos (suggested follow-ups).\n\n"
            f"Interactions:\n{transcript}"
        )
        try:
            # Use brain to get structured summary in JSON format
            response = ask_brain(user_prompt, system_prompt="You are a helpful assistant that summarizes daily chats.", response_format="json")
        except Exception as e:
            logger.error(f"DreamCycle: LLM summarization failed: {e}")
            response = ""
        # Parse LLM response into a dict
        summary_data: Dict[str, Any]
        try:
            summary_data = json.loads(response.strip())
        except Exception:
            # If the assistant didn't return valid JSON, fall back to a basic summary text
            summary_data = {
                "summary": response.strip() or "No summary available.",
                "highlights": [],
                "learnings": [],
                "todos": []
            }
        # Add timestamp and ensure all expected fields exist
        summary_data["timestamp"] = self.now.isoformat()
        summary_data.setdefault("summary", "")
        summary_data.setdefault("highlights", [])
        summary_data.setdefault("learnings", [])
        summary_data.setdefault("todos", [])
        return summary_data
    
    # def run(self) -> str:
    #     t0 = time.time()
    #     logs = self._collect_log_snippets()
    #     summary = self._summarize_patterns(logs)
    #     payload = {
    #         "timestamp": self.now.isoformat(),
    #         "logs_considered": list(logs.keys()),
    #         "summary": summary,
    #         "duration_sec": round(time.time() - t0, 3),
    #     }
    #     out_path = os.path.join(
    #         CFG.learning_dir,
    #         f"dream_summary_{self.now.strftime('%Y%m%d_%H%M%S')}.json",
    #     )
    #     with open(out_path, "w", encoding="utf-8") as f:
    #         json.dump(payload, f, indent=2)
    #     return out_path


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
    
    def write_summary(self, summary_data: Dict[str, Any]) -> str:
        """Write the summary data to a JSON file and insert a summary event into memory."""
        # Determine output file path
        filename = f"dream_summary_{self.now.strftime('%Y%m%d_%H%M%S')}.json"
        out_path = os.path.join("data", "summaries", filename)
        # Write JSON summary to file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"DreamCycle: Summary written to {out_path}")
        # Insert a condensed summary event into MemoryStore
        summary_text = summary_data.get("summary", "(No summary)")
        try:
            # Use high importance to preserve this summary long-term
            self.store.add_event(summary_text, importance=50.0, type_="dream_summary")
            logger.info("DreamCycle: Summary event stored in MemoryStore (type=dream_summary).")
        except Exception as e:
            logger.error(f"DreamCycle: Failed to insert summary event into memory: {e}")
        return out_path