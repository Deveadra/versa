from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from base.self_improve.score_types import ScoreboardRun


def ensure_self_improve_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS repo_score_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          run_type TEXT NOT NULL,
          mode TEXT NOT NULL,
          fix_enabled INTEGER NOT NULL DEFAULT 0,
          git_branch TEXT,
          git_sha TEXT,
          score REAL NOT NULL DEFAULT 0,
          passed INTEGER NOT NULL DEFAULT 0,
          metrics_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS repo_improvement_attempts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          iteration INTEGER NOT NULL,
          baseline_run_id INTEGER NOT NULL,
          before_run_id INTEGER NOT NULL,
          after_run_id INTEGER,
          branch TEXT NOT NULL,
          proposal_title TEXT,
          proposal_json TEXT,
          pr_url TEXT,
          improved INTEGER NOT NULL DEFAULT 0,
          error_text TEXT
        );

        CREATE TABLE IF NOT EXISTS capability_gaps (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          source TEXT NOT NULL,
          fingerprint TEXT NOT NULL UNIQUE,
          requested_capability TEXT NOT NULL,
          observed_failure TEXT,
          classification TEXT NOT NULL,
          repro_steps TEXT,
          priority INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'new',
          metadata_json TEXT
        );
        """
    )
    conn.commit()


def make_gap_fingerprint(*parts: str) -> str:
    s = "|".join((p or "").strip() for p in parts)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def upsert_gap(
    conn: sqlite3.Connection,
    *,
    source: str,
    fingerprint: str,
    requested_capability: str,
    observed_failure: str | None,
    classification: str,
    repro_steps: str | None,
    priority: int = 0,
    status: str = "new",
    metadata: dict[str, Any] | None = None,
) -> int | None:
    ensure_self_improve_schema(conn)
    try:
        cur = conn.execute(
            """
            INSERT INTO capability_gaps(
              source, fingerprint, requested_capability, observed_failure,
              classification, repro_steps, priority, status, metadata_json
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                source,
                fingerprint,
                requested_capability,
                observed_failure,
                classification,
                repro_steps,
                int(priority),
                status,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None


def fetch_open_gaps(conn: sqlite3.Connection, *, limit: int = 5) -> list[dict[str, Any]]:
    ensure_self_improve_schema(conn)
    cur = conn.execute(
        """
        SELECT id, source, fingerprint, requested_capability, observed_failure,
               classification, repro_steps, priority, status, metadata_json
        FROM capability_gaps
        WHERE status IN ('new','queued')
        ORDER BY priority DESC, id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        md = {}
        try:
            md = json.loads(r[9] or "{}")
        except Exception:
            md = {}
        out.append(
            {
                "id": r[0],
                "source": r[1],
                "fingerprint": r[2],
                "requested_capability": r[3],
                "observed_failure": r[4],
                "classification": r[5],
                "repro_steps": r[6],
                "priority": r[7],
                "status": r[8],
                "metadata": md,
            }
        )
    return out


def mark_gap_status(conn: sqlite3.Connection, gap_id: int, status: str) -> None:
    ensure_self_improve_schema(conn)
    conn.execute("UPDATE capability_gaps SET status=? WHERE id=?", (status, int(gap_id)))
    conn.commit()


def insert_score_run(
    conn: sqlite3.Connection,
    *,
    run_type: str,
    run: ScoreboardRun,
    git_branch: str | None,
    git_sha: str | None,
) -> int:
    ensure_self_improve_schema(conn)

    payload = {
        "mode": run.mode,
        "fix_enabled": bool(run.fix_enabled),
        "total_duration_ms": run.total_duration_ms,
        "gates_failing": run.gates_failing,
        "score": run.score(),
        "tool_results": {
            k: {
                "name": v.name,
                "exit_code": v.exit_code,
                "duration_ms": v.duration_ms,
                "stdout_tail": v.stdout_tail,
                "stderr_tail": v.stderr_tail,
                "parsed": v.parsed,
            }
            for k, v in run.tool_results.items()
        },
    }

    cur = conn.execute(
        """
        INSERT INTO repo_score_runs(
          run_type, mode, fix_enabled, git_branch, git_sha, score, passed, metrics_json
        )
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            run_type,
            run.mode,
            1 if run.fix_enabled else 0,
            git_branch,
            git_sha,
            float(run.score()),
            1 if run.passed() else 0,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_attempt(
    conn: sqlite3.Connection,
    *,
    iteration: int,
    baseline_run_id: int,
    before_run_id: int,
    after_run_id: int | None,
    branch: str,
    proposal_title: str | None,
    proposal_json: str | None,
    improved: bool,
    pr_url: str | None = None,
    error_text: str | None = None,
) -> int:
    ensure_self_improve_schema(conn)
    cur = conn.execute(
        """
        INSERT INTO repo_improvement_attempts(
          iteration, baseline_run_id, before_run_id, after_run_id,
          branch, proposal_title, proposal_json, pr_url, improved, error_text
        )
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(iteration),
            int(baseline_run_id),
            int(before_run_id),
            int(after_run_id) if after_run_id is not None else None,
            branch,
            proposal_title,
            proposal_json,
            pr_url,
            1 if improved else 0,
            error_text,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)