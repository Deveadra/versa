from __future__ import annotations

from base.database.sqlite import SQLiteConn


def _cols(conn, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in rows}


def _has_index(conn, table: str, index_name: str) -> bool:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    # PRAGMA index_list returns: seq, name, unique, origin, partial (depending on sqlite ver)
    return any(r[1] == index_name for r in rows)


def test_self_improve_schema_contract(tmp_path):
    db_path = tmp_path / "self_improve_contract.db"

    # SQLiteConn should apply migrations on init (your repo already tracks schema_migrations)
    db = SQLiteConn(str(db_path))
    conn = db.conn

    # ---- repo_score_runs ----
    assert _cols(conn, "repo_score_runs") == {
        "id",
        "created_at",
        "run_type",
        "mode",
        "fix_enabled",
        "git_branch",
        "git_sha",
        "score",
        "passed",
        "metrics_json",
    }

    assert _has_index(conn, "repo_score_runs", "idx_repo_score_runs_created_at")
    assert _has_index(conn, "repo_score_runs", "idx_repo_score_runs_run_type")

    # ---- repo_improvement_attempts ----
    assert _cols(conn, "repo_improvement_attempts") == {
        "id",
        "created_at",
        "iteration",
        "baseline_run_id",
        "before_run_id",
        "after_run_id",
        "branch",
        "proposal_title",
        "proposal_json",
        "pr_url",
        "improved",
        "error_text",
    }

    assert _has_index(conn, "repo_improvement_attempts", "idx_repo_improvement_attempts_created_at")
    assert _has_index(conn, "repo_improvement_attempts", "idx_repo_improvement_attempts_branch")

    # ---- capability_gaps ----
    assert _cols(conn, "capability_gaps") == {
        "id",
        "created_at",
        "source",
        "fingerprint",
        "requested_capability",
        "observed_failure",
        "classification",
        "repro_steps",
        "priority",
        "status",
        "metadata_json",
    }

    assert _has_index(conn, "capability_gaps", "idx_capability_gaps_status_priority")
    assert _has_index(conn, "capability_gaps", "idx_capability_gaps_created_at")
    assert _has_index(conn, "capability_gaps", "idx_capability_gaps_status")
