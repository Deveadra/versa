"""Example bridge for wiring claude_watch_agent into Versa.

Suggested eventual placement inside the repo:
    src/base/research/claude_watch.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_watch_agent import ClaudeWatchAgent, Config


def run_claude_watch_once(
    *,
    output_dir: str = "./reports/claude_watch",
    state_db: str = "./state/claude_watch.db",
) -> dict[str, Any]:
    config = Config(
        output_dir=Path(output_dir),
        state_db_path=Path(state_db),
    )
    agent = ClaudeWatchAgent(config)
    try:
        result = agent.run_once()
        return result
    finally:
        agent.close()


def extract_digest_candidates(
    json_digest_path: str, *, minimum_significance: int = 12
) -> list[dict[str, Any]]:
    payload = json.loads(Path(json_digest_path).read_text(encoding="utf-8"))
    approved: list[dict[str, Any]] = []
    for finding in payload:
        if finding["significance"] < minimum_significance:
            continue
        if finding["category"] == "community_claim" and finding["confidence"] == "low":
            continue
        approved.append(
            {
                "source": finding["source"],
                "category": finding["category"],
                "confidence": finding["confidence"],
                "title": finding["title"],
                "summary": finding["summary"],
                "takeaways": finding.get("takeaways", []),
                "metadata": finding.get("metadata", {}),
            }
        )
    return approved
