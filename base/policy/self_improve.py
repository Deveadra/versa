
from __future__ import annotations
import sqlite3, json

from datetime import datetime
from typing import List, Dict, Any
from loguru import logger

def propose_new_rules(policy) -> List[Dict[str, Any]]:
    """
    Scans recent signal + feedback history.
    Generates candidate rules for uncovered patterns or poorly performing topics.
    Returns a list of rule dicts (not inserted unless you choose to).
    """

    conn = policy.conn
    rules = []

    # === Example heuristic 1: detect signals with high variance ===
    sigs = conn.execute("""
        SELECT name, COUNT(*) as n
        FROM context_signals
        GROUP BY name
        HAVING n > 20
    """).fetchall()

    for s in sigs:
        name = s["name"]

        # Skip if a rule already exists for this signal
        exists = conn.execute(
            "SELECT 1 FROM engagement_rules WHERE condition_json LIKE ?",
            (f"%{name}%",)
        ).fetchone()
        if exists:
            continue

        # Very simple prototype: propose a "gap > 120" rule
        cond = {
            "cond": {"gte": [f"signal:{name}", 120]},
            "severity": {"between": [f"signal:{name}", 90, 240]},
            "bindings": {f"{name}_val": f"signal:{name}"}
        }
        tone = {
            "map": [
                {"gte": ["severity", 0.7], "tone": "firm"},
                {"gte": ["severity", 0.4], "tone": "persistent"}
            ],
            "default": "gentle"
        }

        rules.append({
            "name": f"auto_{name}_gap",
            "topic_id": name,
            "priority": 60,
            "cooldown_seconds": 3600,
            "max_per_day": 4,
            "condition": cond,
            "tone_strategy": tone,
            "context_template": f"{name}={{ {name}_val }}"
        })

    # === Example heuristic 2: boost poorly performing rules ===
    bad_rules = conn.execute("""
        SELECT r.id, r.name, r.topic_id, s.ema_success, s.ema_negative
        FROM engagement_rules r
        LEFT JOIN rule_stats s ON s.rule_id=r.id
        WHERE s.ema_success < 0.3 AND s.ema_negative > 0.7
    """).fetchall()

    for r in bad_rules:
        logger.info(f"Rule {r['name']} underperforming. Suggesting alternative.")
        # Propose a modified rule with lower frequency
        rules.append({
            "name": f"retry_{r['name']}",
            "topic_id": r["topic_id"],
            "priority": r.get("priority", 60) + 10,
            "cooldown_seconds": 7200,
            "max_per_day": 2,
            "condition": {"cond":{"exists": f"signal:{r['topic_id']}"}},
            "tone_strategy": {"default":"gentle"},
            "context_template": f"{r['topic_id']} triggered"
        })

    return rules

def _log_audit(conn: sqlite3.Connection, rule: Dict[str,Any], rationale: str, details: dict):
    conn.execute("""
        INSERT INTO rule_audit(rule_name, topic_id, rationale, details_json)
        VALUES(?,?,?,?)
    """, (rule["name"], rule["topic_id"], rationale, json.dumps(details)))
    conn.commit()

def propose_new_rules(policy) -> List[Dict[str, Any]]:
    conn = policy.conn
    rules: List[Dict[str, Any]] = []

    # === Heuristic 1: signals with high variance and no rule ===
    sigs = conn.execute("""
        SELECT name, COUNT(*) as n
        FROM context_signals
        GROUP BY name
        HAVING n > 20
    """).fetchall()

    for s in sigs:
        name = s["name"]

        exists = conn.execute(
            "SELECT 1 FROM engagement_rules WHERE condition_json LIKE ?",
            (f"%{name}%",)
        ).fetchone()
        if exists:
            continue

        cond = {
            "cond": {"gte": [f"signal:{name}", 120]},
            "severity": {"between": [f"signal:{name}", 90, 240]},
            "bindings": {f"{name}_val": f"signal:{name}"}
        }
        tone = {
            "map": [
                {"gte": ["severity", 0.7], "tone": "firm"},
                {"gte": ["severity", 0.4], "tone": "persistent"}
            ],
            "default": "gentle"
        }

        rule = {
            "name": f"auto_{name}_gap",
            "topic_id": name,
            "priority": 60,
            "cooldown_seconds": 3600,
            "max_per_day": 4,
            "condition": cond,
            "tone_strategy": tone,
            "context_template": f"{name}={{ {name}_val }}"
        }

        rationale = f"I observed frequent changes in signal '{name}' without an engagement rule. I created a rule to nudge when {name} exceeds 120."
        _log_audit(conn, rule, rationale, {"signal": name, "observations": s["n"]})
        rules.append(rule)

    # === Heuristic 2: underperforming rules (EMA-based) ===
    bad_rules = conn.execute("""
        SELECT r.id, r.name, r.topic_id, s.ema_success, s.ema_negative
        FROM engagement_rules r
        LEFT JOIN rule_stats s ON s.rule_id=r.id
        WHERE s.ema_success < 0.3 AND s.ema_negative > 0.7
    """).fetchall()

    for r in bad_rules:
        new_rule = {
            "name": f"retry_{r['name']}",
            "topic_id": r["topic_id"],
            "priority": (r.get("priority") or 60) + 10,
            "cooldown_seconds": 7200,
            "max_per_day": 2,
            "condition": {"cond": {"exists": f"signal:{r['topic_id']}"}},
            "tone_strategy": {"default": "gentle"},
            "context_template": f"{r['topic_id']} triggered"
        }

        rationale = (
            f"Rule '{r['name']}' for topic '{r['topic_id']}' showed low success (ema_success={r['ema_success']}) "
            f"and high negatives (ema_negative={r['ema_negative']}). I proposed a gentler, less frequent retry rule."
        )
        _log_audit(conn, new_rule, rationale, {"bad_rule": dict(r)})
        rules.append(new_rule)

    # === Heuristic 3: time-of-day failure patterns from rule_history ===
    history = conn.execute("""
        SELECT topic_id, outcome, strftime('%H', fired_at) as hour, COUNT(*) as n
        FROM rule_history
        WHERE fired_at > datetime('now','-7 days')
        GROUP BY topic_id, outcome, hour
    """).fetchall()

    for h in history:
        if h["outcome"] in ("ignore","angry") and int(h["hour"]) >= 21:
            # User ignores this topic at night → propose evening-specific softer rule
            topic = h["topic_id"]
            rule = {
                "name": f"{topic}_evening_soft",
                "topic_id": topic,
                "priority": 70,
                "cooldown_seconds": 10800,
                "max_per_day": 2,
                "condition": {
                    "cond": {
                        "all": [
                            {"eq": ["signal:hour_of_day", int(h["hour"])]},
                            {"exists": f"signal:{topic}"}
                        ]
                    }
                },
                "tone_strategy": {"default": "gentle"},
                "context_template": f"{topic} evening reminder"
            }
            rationale = (
                f"In the last 7 days, topic '{topic}' was ignored or rejected {h['n']} times "
                f"during hour {h['hour']}. I created a gentler, evening-specific rule to adapt."
            )
            _log_audit(conn, rule, rationale, {"history": dict(h)})
            rules.append(rule)

    return rules

def insert_proposed_rules(conn: sqlite3.Connection, rules: List[Dict[str,Any]]):
    for r in rules:
        conn.execute("""
            INSERT INTO engagement_rules
              (name, topic_id, priority, cooldown_seconds, max_per_day,
               condition_json, tone_strategy_json, context_template, enabled)
            VALUES (?,?,?,?,?,?,?,?,1)
            ON CONFLICT(name) DO NOTHING
        """, (
            r["name"],
            r["topic_id"],
            r.get("priority",60),
            r.get("cooldown_seconds",1800),
            r.get("max_per_day",4),
            json.dumps(r["condition"]),
            json.dumps(r["tone_strategy"]),
            r.get("context_template",""),
        ))
    conn.commit()
