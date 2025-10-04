# base/learning/self_improve_runner.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from base.database.sqlite import SQLiteConn
from base.learning import dream_cycle, review
from base.learning.habit_miner import HabitMiner
from base.memory.store import MemoryStore
from base.policy import policy_store, self_improve
from base.policy.policy_store import PolicyStore  # <-- for policy.conn
from base.voice.tts_elevenlabs import Voice
from config.config import settings

LOG_DIR = Path("logs/morning_reviews")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_policy_store() -> policy_store.PolicyStore:
    # Create or reuse your SQLite connection
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return policy_store.PolicyStore(conn)


def review_proposal(conn, rule_id: int, action: str):
    if action not in ("approve", "deny", "revert"):
        raise ValueError("Invalid action")

    if action == "approve":
        conn.execute("UPDATE proposed_rules SET status='approved' WHERE id=?", (rule_id,))
    elif action == "deny":
        conn.execute("UPDATE proposed_rules SET status='denied' WHERE id=?", (rule_id,))
    elif action == "revert":
        # find applied rule, delete it from engagement_rules, mark as reverted
        rule = conn.execute("SELECT * FROM proposed_rules WHERE id=?", (rule_id,)).fetchone()
        if rule:
            conn.execute("DELETE FROM engagement_rules WHERE name=?", (rule["name"],))
            conn.execute("UPDATE proposed_rules SET status='reverted' WHERE id=?", (rule_id,))
    conn.commit()


def accept_proposal(rule_id: int):
    db = SQLiteConn(settings.db_path)
    conn = db.conn

    p = conn.execute("SELECT * FROM proposed_rules WHERE id=?", (rule_id,)).fetchone()
    if not p:
        logger.error(f"No proposal with id={rule_id}")
        return

    # Insert into engagement_rules
    conn.execute(
        """
        INSERT INTO engagement_rules
          (name, topic_id, priority, cooldown_seconds, max_per_day,
           condition_json, tone_strategy_json, context_template, enabled)
        VALUES (?,?,?,?,?,?,?,?,1)
        ON CONFLICT(name) DO NOTHING
        """,
        (
            p["name"],
            p["topic_id"],
            p["priority"],
            p["cooldown_seconds"],
            p["max_per_day"],
            p["condition_json"],
            p["tone_strategy_json"],
            p["context_template"],
        ),
    )
    conn.execute("UPDATE proposed_rules SET status='accepted' WHERE id=?", (rule_id,))
    conn.commit()
    logger.info(f"Accepted proposal {p['name']} for topic {p['topic_id']}.")


def reject_proposal(rule_id: int):
    db = SQLiteConn(settings.db_path)
    conn = db.conn
    conn.execute("UPDATE proposed_rules SET status='rejected' WHERE id=?", (rule_id,))
    conn.commit()
    logger.info(f"Rejected proposal id={rule_id}.")


def revert_proposal(rule_id: int):
    db = SQLiteConn(settings.db_path)
    conn = db.conn

    p = conn.execute(
        "SELECT * FROM proposed_rules WHERE id=? AND status='accepted'", (rule_id,)
    ).fetchone()
    if not p:
        logger.error(f"No accepted proposal with id={rule_id} to revert.")
        return

    conn.execute("DELETE FROM engagement_rules WHERE name=?", (p["name"],))
    conn.execute("UPDATE proposed_rules SET status='reverted' WHERE id=?", (rule_id,))
    conn.commit()
    logger.info(f"Reverted proposal {p['name']} for topic {p['topic_id']}.")


def morning_review(text_only: bool = False):
    db = SQLiteConn(settings.db_path)
    conn = db.conn

    proposals = conn.execute(
        "SELECT * FROM proposed_rules WHERE status='pending' ORDER BY created_at"
    ).fetchall()

    if not proposals:
        msg = "No proposals waiting for review."
        if not text_only:
            Voice.get_instance().speak_async(msg)
        logger.info(msg)
        return msg

    # --- Build summary text ---
    lines = []
    header = f"Morning Review {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for p in proposals:
        lines.append(
            f"[{p['id']}] Rule '{p['name']}' for topic '{p['topic_id']}' "
            f"(confidence={p['confidence']:.2f}, score={p['score']:.2f})"
        )
        if p["rationale"]:
            lines.append(f"   ↳ {p['rationale']}")

    summary_text = "\n".join(lines)

    # --- Save to logs ---
    fname = LOG_DIR / f"morning_review_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    logger.info(f"Morning Review saved to {fname}")
    if text_only:
        return summary_text

    # --- Voice ---
    Voice.get_instance().speak_async(
        f"You have {len(proposals)} new rule proposals. "
        f"First is {proposals[0]['name']} for {proposals[0]['topic_id']}."
    )

    return summary_text


# # DAY AND NIGHT CYCLE
# def run_self_improve():
#     store = get_policy_store()
#     proposals = self_improve.propose_new_rules(store)

#     if not proposals:
#         logger.info("No new self-improvement proposals today.")
#         return

#     accepted = [r for r in proposals if r.get("confidence", 0.65) >= 0.65]
#     if not accepted:
#         logger.info("All proposals below threshold; skipping insert.")
#         return

#     explanations = []
#     for rule in accepted:
#         expl = (
#             f"I detected a pattern in '{rule['topic_id']}'. "
#             f"The rule '{rule['name']}' should help adapt. "
#             f"My confidence is {rule['confidence']:.2f}."
#         )
#         explanations.append(expl)

#     Voice.get_instance().speak_async(
#         f"I have created {len(accepted)} new rules to improve our interactions. "
#         "You can review my reasoning in the logs."
#     )

#     logger.info("Ultron’s self-improvement proposals:")
#     for expl in explanations:
#         logger.info("  " + expl)

#     self_improve.insert_proposed_rules(store.conn, accepted)


def run_nightly_self_improve():
    conn = SQLiteConn(settings.db_path).conn
    policy = PolicyStore(conn)

    # 1) (optional) apply reverts first if you added that helper
    # from base.learning.self_improve_runner import apply_reverts
    # apply_reverts(conn)

    # 2) heuristic proposals → queue (pending)
    heuristics = self_improve.propose_new_rules(policy)
    heuristics = [r for r in heuristics if r.get("confidence", 0.6) >= 0.6]
    if heuristics:
        for r in heuristics:
            # queue proposals into proposed_rules instead of live table
            conn.execute(
                """
                INSERT INTO proposed_rules
                (name, topic_id, condition_json, tone_strategy_json, priority, cooldown_seconds,
                 max_per_day, context_template, confidence, score, rationale, status, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?, 'pending', datetime('now'))
                ON CONFLICT(name) DO NOTHING
                """,
                (
                    r["name"],
                    r["topic_id"],
                    json.dumps(r["condition"]),
                    json.dumps(r["tone_strategy"]),
                    r.get("priority", 60),
                    r.get("cooldown_seconds", 1800),
                    r.get("max_per_day", 4),
                    r.get("context_template", ""),
                    r.get("confidence", 0.6),
                    r.get("score", 0.6),
                    r.get("rationale", "auto-proposed"),
                ),
            )
        conn.commit()
        logger.info(f"Queued {len(heuristics)} heuristic proposals (pending).")
    else:
        logger.info("No heuristic proposals tonight.")

    # 3) GPT-based proposals/refinements
    dream_cycle.propose_new_signals_and_rules(conn)
    dream_cycle.expand_consequence_map(conn)
    dream_cycle.cluster_complaints(conn)

    # 4) gentle reminder in logs + voice
    pending = review.list_pending(conn)
    if pending:
        logger.info("Review pending proposals: uv run python -m assistant.cli.review_cli list")
        Voice.get_instance().speak_async(
            "I drafted new proposals. Please review them when you have time."
        )
    else:
        logger.info("No pending proposals this cycle.")
        Voice.get_instance().speak_async("No new proposals tonight. All stable.")


def run_morning_review_digest():
    """Runs once each morning to drop the text file + optional voice cue."""
    conn = SQLiteConn(settings.db_path).conn
    path = review.write_morning_digest(conn, speak=True)
    logger.info(f"Morning review written to {path}")


# UNIFIED NIGHT CYCLE
# ---- 0. setup connection + stores ----
def run_unified_night_cycle(auto_insert: bool = False):
    db = SQLiteConn(settings.db_path)
    memory_store = MemoryStore(db)
    policy = PolicyStore(db.conn)
    conn = policy.conn

    apply_reverts(conn)

    # ---- 1. Persona update ----
    habit_miner = HabitMiner(db, memory_store, memory_store)
    profile = habit_miner.mine()
    logger.info("HabitMiner refreshed persona before nightly rules.")

    # ---- 2. Heuristic proposals ----
    heuristics = self_improve.propose_new_rules(policy)
    heuristics = [r for r in heuristics if r.get("confidence", 0.6) >= 0.6]

    if heuristics:
        if auto_insert:
            self_improve.insert_proposed_rules(conn, heuristics)
            logger.info(f"Auto-inserted {len(heuristics)} heuristic rules.")
            Voice.get_instance().speak_async(
                f"I’ve created {len(heuristics)} new rules tonight to improve our interactions."
            )
        else:
            for r in heuristics:
                conn.execute(
                    """
                    INSERT INTO proposed_rules
                      (name, topic_id, priority, cooldown_seconds, max_per_day,
                       condition_json, tone_strategy_json, context_template,
                       confidence, score, created_at, rationale, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),?,'pending')
                    """,
                    (
                        r["name"],
                        r["topic_id"],
                        r.get("priority", 60),
                        r.get("cooldown_seconds", 1800),
                        r.get("max_per_day", 4),
                        json.dumps(r["condition"]),
                        json.dumps(r["tone_strategy"]),
                        r.get("context_template", ""),
                        r.get("confidence", 0.0),
                        r.get("score", 0.0),
                        f"Queued for review: {r.get('source','unknown')}",
                    ),
                )
            conn.commit()
            logger.info(f"Queued {len(heuristics)} heuristic rules for review.")
            Voice.get_instance().speak_async(
                f"I’ve drafted {len(heuristics)} new rules, but I’ll wait for your review."
            )
    else:
        logger.info("No strong heuristic rules tonight.")
        Voice.get_instance().speak_async(
            "No strong heuristic rules tonight, but I refreshed signals and complaints."
        )

    # ---- 3. Dream cycle (LLM proposals) ----
    dream_cycle.propose_new_signals_and_rules(conn)
    dream_cycle.expand_consequence_map(conn)
    dream_cycle.cluster_complaints(conn)


def start_scheduler(auto_insert: bool = False):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: run_unified_night_cycle(auto_insert=auto_insert), "cron", hour=2, minute=0
    )

    # Morning digest — configurable
    hr = getattr(settings, "morning_review_hour", 8)
    mn = getattr(settings, "morning_review_minute", 30)
    scheduler.add_job(run_morning_review_digest, "cron", hour=hr, minute=mn)

    scheduler.start()
    logger.info(f"Unified scheduler started (nightly=02:00, morning={hr:02d}:{mn:02d}).")


def apply_reverts(conn):
    """
    Remove any rules marked as reverted from engagement_rules.
    Archive them in proposed_rules for history.
    """
    reverted = conn.execute("SELECT name FROM proposed_rules WHERE status='reverted'").fetchall()

    for r in reverted:
        logger.info(f"Applying revert: removing {r['name']} from engagement_rules")
        conn.execute("DELETE FROM engagement_rules WHERE name=?", (r["name"],))
        conn.execute(
            """
            UPDATE proposed_rules
            SET status='archived'
            WHERE name=? AND status='reverted'
        """,
            (r["name"],),
        )
        conn.execute(
            """
            INSERT INTO audit_log (created_at, rationale)
            VALUES (datetime('now'), ?)
        """,
            (f"Removed reverted rule '{r['name']}' from engagement_rules",),
        )

    conn.commit()
