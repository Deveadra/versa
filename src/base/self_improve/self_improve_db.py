from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_gap_fingerprint(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _has_table(conn, name: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    row = cur.fetchone()
    return bool(row)


def ensure_self_improve_schema(conn) -> None:
    """
    Runtime safety net ONLY.
    The authoritative schema lives in SQL migrations.
    If a DB is missing the self-improve tables (fresh / corrupted / tests), apply the migration SQL.
    """
    required = ("repo_score_runs", "repo_improvement_attempts", "capability_gaps")
    if all(_has_table(conn, t) for t in required):
        return

    migrations_dir = Path(__file__).resolve().parents[1] / "database" / "migrations"
    candidates = ("0002_self_improve.sql", "0002_repo_janitor.sql")

    sql = None
    for name in candidates:
        p = migrations_dir / name
        if p.exists():
            sql = p.read_text(encoding="utf-8", errors="ignore")
            break

    if not sql:
        raise RuntimeError(
            f"Self-improve schema missing and no migration file found in {migrations_dir}"
        )

    conn.executescript(sql)
    conn.commit()


def insert_score_run(
    conn,
    *,
    run_type: str,
    mode: str,
    fix_enabled: bool,
    git_branch: str | None,
    git_sha: str | None,
    score: float,
    passed: bool,
    metrics: dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO repo_score_runs(run_type, mode, fix_enabled, git_branch, git_sha, score, passed, metrics_json)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_type,
            mode,
            1 if fix_enabled else 0,
            git_branch,
            git_sha,
            float(score),
            1 if passed else 0,
            json.dumps(metrics, ensure_ascii=False),
        ),
    )
    conn.commit()
    rid = cur.lastrowid
    if rid is None:
        raise RuntimeError("insert_score_run: lastrowid is None")
    return int(rid)


def insert_improvement_attempt(
    conn,
    *,
    iteration: int,
    baseline_run_id: int,
    before_run_id: int,
    after_run_id: int | None,
    branch: str,
    proposal_title: str | None,
    proposal_json: dict[str, Any] | None,
    pr_url: str | None,
    improved: bool,
    error_text: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO repo_improvement_attempts(
          iteration, baseline_run_id, before_run_id, after_run_id,
          branch, proposal_title, proposal_json, pr_url, improved, error_text
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(iteration),
            int(baseline_run_id),
            int(before_run_id),
            int(after_run_id) if after_run_id is not None else None,
            branch,
            proposal_title,
            json.dumps(proposal_json, ensure_ascii=False) if proposal_json is not None else None,
            pr_url,
            1 if improved else 0,
            error_text,
        ),
    )
    conn.commit()
    rid = cur.lastrowid
    if rid is None:
        raise RuntimeError("insert_improvement_attempt: lastrowid is None")
    return int(rid)


def upsert_gap(
    conn,
    *,
    source: str,
    fingerprint: str,
    requested_capability: str,
    observed_failure: str | None,
    classification: str,
    repro_steps: str | None,
    priority: int,
    status: str = "queued",
    metadata: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO capability_gaps(
          source, fingerprint, requested_capability, observed_failure,
          classification, repro_steps, priority, status, metadata_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
          requested_capability=excluded.requested_capability,
          classification=excluded.classification,
          repro_steps=COALESCE(excluded.repro_steps, capability_gaps.repro_steps),
          priority=MAX(capability_gaps.priority, excluded.priority),
          observed_failure=COALESCE(excluded.observed_failure, capability_gaps.observed_failure),
          metadata_json=excluded.metadata_json,
          status=CASE
              WHEN capability_gaps.status = 'fixed'
                   AND excluded.status IN ('queued', 'new', 'in_progress')
              THEN capability_gaps.status
              WHEN capability_gaps.status = 'in_progress'
                   AND excluded.status IN ('queued', 'new')
              THEN capability_gaps.status
              ELSE excluded.status
          END
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


def fetch_open_gaps(conn, *, limit: int = 5) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, source, fingerprint, requested_capability, observed_failure,
               classification, repro_steps, priority, status
        FROM capability_gaps
        WHERE status IN ('queued','in_progress','new')
        ORDER BY priority DESC, id ASC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    cols = [d[0] for d in (cur.description or [])]
    if not cols:
        return []
    return [dict(zip(cols, row)) for row in rows]


def mark_gap_status(conn, gap_id: int, status: str) -> None:
    conn.execute("UPDATE capability_gaps SET status=? WHERE id=?", (status, int(gap_id)))
    conn.commit()
