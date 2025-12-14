# assistant/base/learning/review.py
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from base.voice.tts_elevenlabs import Voice

LOG_DIR = Path("logs/morning_reviews")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _format_pending_lines(rows):
    lines = []
    for p in rows:
        conf = (
            f"{float(p.get('confidence', 0.0)):.2f}" if p.get("confidence") is not None else "n/a"
        )
        score = f"{float(p.get('score', 0.0)):.2f}" if p.get("score") is not None else "n/a"
        lines.append(
            f"[{p['id']}] {p['name']}  topic={p['topic_id']}  (confidence={conf}, score={score})"
        )
        if p.get("rationale"):
            lines.append(f"   ↳ {p['rationale']}")
    return lines


def write_morning_digest(conn, speak: bool = True) -> str:
    """
    Create/save the morning review .txt and optionally speak a short cue.
    Includes both pending and approved-but-not-yet-applied proposals.
    """
    pending = list_pending(conn)
    approved = conn.execute(
        "SELECT * FROM proposed_rules WHERE status='approved' ORDER BY created_at"
    ).fetchall()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Morning Review {timestamp}"
    sep = "=" * len(header)

    sections = []

    # Pending
    if pending:
        body = "\n".join(_format_pending_lines(pending))
        sections.append(f"--- Pending Proposals ---\n{body}")
    else:
        sections.append("--- Pending Proposals ---\n(none)")

    # Approved (queued for tonight)
    if approved:
        lines = []
        for r in approved:
            conf = f"{float(r['confidence']):.2f}" if r.get("confidence") else "n/a"
            score = f"{float(r['score']):.2f}" if r.get("score") else "n/a"
            lines.append(
                f"[{r['id']}] {r['name']} topic={r['topic_id']} (confidence={conf}, score={score})"
            )
            if r.get("rationale"):
                lines.append(f"   ↳ {r['rationale']}")
        sections.append("--- Approved (queued for tonight) ---\n" + "\n".join(lines))
    else:
        sections.append("--- Approved (queued for tonight) ---\n(none)")

    # Assemble
    text = f"{sep}\n{header}\n{sep}\n\n" + "\n\n".join(sections)
    text += "\n\nTip: approve/deny with CLI → uv run python -m assistant.cli.review_cli list|approve|deny|revert\n"

    # Save
    fname = LOG_DIR / f"morning_review_{datetime.now().strftime('%Y%m%d')}.txt"
    fname.write_text(text, encoding="utf-8")

    # Voice cue
    if speak:
        if pending:
            Voice.get_instance().speak_async(
                f"You have {len(pending)} new rule proposals pending review. "
                "Some approved rules are queued for tonight."
            )
        elif approved:
            Voice.get_instance().speak_async(
                f"You have {len(approved)} approved rules queued for tonight."
            )
        else:
            Voice.get_instance().speak_async("No new proposals or queued rules this morning.")

    return str(fname)


def summarize_pending_text(conn) -> str:
    """Return the pending proposals as a single text blob (for UI/DM)."""
    rows = list_pending(conn)
    if not rows:
        return "No proposals waiting for review."
    return "\n".join(_format_pending_lines(rows))


def list_pending(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all pending proposals (status='pending')."""
    cur = conn.execute("SELECT * FROM proposed_rules WHERE status='pending'")
    return [dict(r) for r in cur.fetchall()]


def approve_rule(conn: sqlite3.Connection, name: str) -> None:
    """Mark a proposal as approved for next nightly cycle."""
    conn.execute(
        """
        UPDATE proposed_rules
        SET status='approved', approved_at=datetime('now')
        WHERE name=? AND status='pending'
    """,
        (name,),
    )
    conn.execute(
        """
        INSERT INTO audit_log (created_at, rationale)
        VALUES (datetime('now'), ?)
    """,
        (f"Rule '{name}' approved by user"),
    )
    conn.commit()
    logger.info(f"Approved rule '{name}'.")


def deny_rule(conn: sqlite3.Connection, name: str) -> None:
    """Deny a proposal permanently."""
    conn.execute(
        """
        UPDATE proposed_rules
        SET status='denied', denied_at=datetime('now')
        WHERE name=? AND status='pending'
    """,
        (name,),
    )
    conn.execute(
        """
        INSERT INTO audit_log (created_at, rationale)
        VALUES (datetime('now'), ?)
    """,
        (f"Rule '{name}' denied by user"),
    )
    conn.commit()
    logger.info(f"Denied rule '{name}'.")


def revert_rule(conn: sqlite3.Connection, name: str) -> None:
    """Schedule a rule to be removed on next nightly cycle."""
    conn.execute(
        """
        UPDATE proposed_rules
        SET status='reverted', reverted_at=datetime('now')
        WHERE name=? AND status!='pending'
    """,
        (name,),
    )
    conn.execute(
        """
        INSERT INTO audit_log (created_at, rationale)
        VALUES (datetime('now'), ?)
    """,
        (f"Rule '{name}' marked for revert by user"),
    )
    conn.commit()
    logger.info(f"Rule '{name}' scheduled for revert.")


def list_history(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    """Show the last N proposals (any status)."""
    cur = conn.execute("SELECT * FROM proposed_rules ORDER BY rowid DESC LIMIT ?", (limit,))
    return [dict(r) for r in cur.fetchall()]
