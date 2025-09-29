# assistant/cli/review_cli.py
from __future__ import annotations
import argparse
from tabulate import tabulate

from base.learning import review
from base.database.sqlite import SQLiteConn
from config.config import settings


REMINDER = """
Available commands:
  uv run python -m assistant.cli.review_cli list
  uv run python -m assistant.cli.review_cli list-approved
  uv run python -m assistant.cli.review_cli approve <rule_name>
  uv run python -m assistant.cli.review_cli deny <rule_name>
  uv run python -m assistant.cli.review_cli revert <rule_name>
  uv run python -m assistant.cli.review_cli history
  uv run python -m assistant.cli.review_cli commands
"""


def _print_reminder():
    print(REMINDER.strip())


def _connect():
    # Normalize to raw sqlite3 connection
    return SQLiteConn(settings.db_path).conn


def _fmt_conf(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"


def cmd_list(conn):
    rows = review.list_pending(conn)
    if not rows:
        print("✅ No pending proposals.")
    else:
        print(tabulate(
            [(r["id"], r["name"], r["topic_id"], _fmt_conf(r.get("confidence")), _fmt_conf(r.get("score"))) for r in rows],
            headers=["ID", "Name", "Topic", "Confidence", "Score"]
        ))
    _print_reminder()


def cmd_list_approved(conn):
    cur = conn.execute(
        "SELECT * FROM proposed_rules WHERE status='approved' ORDER BY approved_at NULLS LAST, created_at"
        if "NULLS" in conn.execute("SELECT 'x'").fetchone()[0]  # naive guard; SQLite doesn't support NULLS LAST
        else "SELECT * FROM proposed_rules WHERE status='approved' ORDER BY COALESCE(approved_at, created_at)"
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        print("✅ No approved proposals queued for tonight.")
    else:
        print(tabulate(
            [(r["id"], r["name"], r["topic_id"], _fmt_conf(r.get("confidence")), _fmt_conf(r.get("score"))) for r in rows],
            headers=["ID", "Name", "Topic", "Confidence", "Score"]
        ))
    _print_reminder()


def cmd_history(conn):
    rows = review.list_history(conn)
    if not rows:
        print("No proposal history yet.")
    else:
        print(tabulate(
            [(r["id"], r["name"], r["topic_id"], r["status"]) for r in rows],
            headers=["ID", "Name", "Topic", "Status"]
        ))
    _print_reminder()


def cmd_approve(conn, name: str):
    review.approve_rule(conn, name)
    print(f"✅ Approved: {name}")
    _print_reminder()


def cmd_deny(conn, name: str):
    review.deny_rule(conn, name)
    print(f"❌ Denied: {name}")
    _print_reminder()


def cmd_revert(conn, name: str):
    review.revert_rule(conn, name)
    print(f"↩️  Scheduled revert: {name}")
    _print_reminder()


def main():
    parser = argparse.ArgumentParser(description="Ultron rule review CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List pending proposals")
    sub.add_parser("list-approved", help="List approved proposals queued for tonight")
    sub.add_parser("history", help="Show recent proposals (any status)")
    sub.add_parser("commands", help="Show command cheat-sheet")

    ap = sub.add_parser("approve", help="Approve a proposal by name")
    ap.add_argument("name")

    dp = sub.add_parser("deny", help="Deny a proposal by name")
    dp.add_argument("name")

    rp = sub.add_parser("revert", help="Revert a live/approved rule by name")
    rp.add_argument("name")

    args = parser.parse_args()
    conn = _connect()

    if args.cmd == "list":
        cmd_list(conn)
    elif args.cmd == "list-approved":
        cmd_list_approved(conn)
    elif args.cmd == "approve":
        cmd_approve(conn, args.name)
    elif args.cmd == "deny":
        cmd_deny(conn, args.name)
    elif args.cmd == "revert":
        cmd_revert(conn, args.name)
    elif args.cmd == "history":
        cmd_history(conn)
    elif args.cmd == "commands":
        _print_reminder()
    else:
        _print_reminder()


if __name__ == "__main__":
    main()
