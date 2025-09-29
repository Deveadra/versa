# base/learning/self_improve.py
from __future__ import annotations
import sqlite3, json, math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Iterable
from loguru import logger


# ---------- helpers ----------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return bool(row)

def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    # sqlite3.Row is mapping-like but doesn't have .get()
    try:
        return row[key]
    except Exception:
        return default

def _log_audit(conn: sqlite3.Connection, rule: Dict[str, Any], rationale: str, details: dict) -> None:
    conn.execute(
        """
        INSERT INTO rule_audit(rule_name, topic_id, rationale, details_json, created_at)
        VALUES(?,?,?,?,datetime('now'))
        """,
        (rule["name"], rule["topic_id"], rationale, json.dumps(details)),
    )
    conn.commit()

def _canonical_rule_name(prefix: str, topic_id: str) -> str:
    return f"{prefix}_{topic_id}".replace(" ", "_").lower()

def _similar_rule_exists(conn: sqlite3.Connection, topic_id: str, needle: str) -> bool:
    # Very light de-dupe: avoid proposing a rule if another mentions same topic & operator already
    row = conn.execute(
        "SELECT 1 FROM engagement_rules WHERE topic_id=? AND condition_json LIKE ? LIMIT 1",
        (topic_id, f"%{needle}%"),
    ).fetchone()
    return bool(row)

def _mean_std(vals: Iterable[float]) -> tuple[float, float]:
    vals = list(vals)
    if not vals:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
    return mu, math.sqrt(var)


# ---------- core ----------

def propose_new_rules(policy) -> List[Dict[str, Any]]:
    """
    Scan signal + feedback history and propose new/adjusted rules.
    Returns a list of **ranked** rule dicts (each includes 'score' and 'confidence').
    Nothing is inserted here—use insert_proposed_rules() to persist.
    """
    conn: sqlite3.Connection = policy.conn  # policy.conn should be raw sqlite3.Connection
    proposals: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    # === Heuristic 1: high-variance / frequently changing signals ===
    # Prefer true history if available (signal_log: name, value, ts); else fall back to presence in context_signals.
    if _table_exists(conn, "signal_log"):
        # Look at the last 50 points per signal
        sig_names = [r["name"] for r in conn.execute(
            "SELECT DISTINCT name FROM signal_log WHERE ts > datetime('now','-14 days')"
        ).fetchall()]

        for name in sig_names:
            rows = conn.execute(
                """
                SELECT value FROM signal_log
                WHERE name=? AND ts > datetime('now','-14 days')
                ORDER BY ts DESC LIMIT 50
                """,
                (name,),
            ).fetchall()
            values = []
            for r in rows:
                try:
                    values.append(float(_row_get(r, "value", 0.0)))
                except Exception:
                    continue

            if len(values) < 15:
                continue  # need some history

            mu, sd = _mean_std(values)
            if sd < 1e-6:
                continue

            # propose threshold ~= mean + 1.5*std (spike detection)
            thr = mu + 1.5 * sd
            cond = {
                "cond": {"gt": [f"signal:{name}", round(thr, 2)]},
                "severity": {"between": [f"signal:{name}", round(mu + 0.5 * sd, 2), round(mu + 2.0 * sd, 2)]},
                "bindings": {f"{name}_val": f"signal:{name}"},
            }
            tone = {
                "map": [
                    {"gte": ["severity", 0.75], "tone": "firm"},
                    {"gte": ["severity", 0.45], "tone": "persistent"},
                ],
                "default": "gentle",
            }

            rule = {
                "name": _canonical_rule_name("auto_spike", name),
                "topic_id": name,
                "priority": 65,
                "cooldown_seconds": 3600,
                "max_per_day": 4,
                "condition": cond,
                "tone_strategy": tone,
                "context_template": f"{name}={{ {name}_val }}",
                "source": "variance_spike",
                "confidence": min(0.95, 0.55 + min(0.4, sd / (abs(mu) + 1e-6))),
                "score": min(1.0, 0.6 + min(0.3, sd / (abs(mu) + 1e-6))),
            }
            if _similar_rule_exists(conn, name, '"gt"') or rule["name"] in seen_names:
                continue
            _log_audit(conn, rule, f"High variance for '{name}' (μ={mu:.2f}, σ={sd:.2f}); proposing spike rule.", {"mu": mu, "sd": sd})
            seen_names.add(rule["name"])
            proposals.append(rule)

    else:
        # fallback: just look for signals that appear often in the table
        # (not a variance check, but still identifies active signals with no rules)
        sigs = conn.execute(
            "SELECT name, COUNT(*) AS n FROM context_signals GROUP BY name HAVING n > 20"
        ).fetchall()
        for s in sigs:
            name = s["name"]
            if _similar_rule_exists(conn, name, f"signal:{name}"):
                continue

            cond = {
                "cond": {"gte": [f"signal:{name}", 120]},
                "severity": {"between": [f"signal:{name}", 90, 240]},
                "bindings": {f"{name}_val": f"signal:{name}"},
            }
            tone = {
                "map": [
                    {"gte": ["severity", 0.7], "tone": "firm"},
                    {"gte": ["severity", 0.4], "tone": "persistent"},
                ],
                "default": "gentle",
            }
            rule = {
                "name": _canonical_rule_name("auto_gap", name),
                "topic_id": name,
                "priority": 60,
                "cooldown_seconds": 3600,
                "max_per_day": 4,
                "condition": cond,
                "tone_strategy": tone,
                "context_template": f"{name}={{ {name}_val }}",
                "source": "presence_only",
                "confidence": 0.5,
                "score": 0.55,
            }
            _log_audit(conn, rule, f"Frequent updates for '{name}' (no variance data). Proposed simple gap rule.", {"name": name, "observations": _row_get(s, "n", 0)})
            seen_names.add(rule["name"])
            proposals.append(rule)

    # === Heuristic 2: under-performing rules (EMA) → propose gentler, less frequent variant ===
    bad_rules = conn.execute(
        """
        SELECT r.id, r.name, r.topic_id,
               COALESCE(s.ema_success, 0.5) AS ema_success,
               COALESCE(s.ema_negative, 0.5) AS ema_negative
        FROM engagement_rules r
        LEFT JOIN rule_stats s ON s.rule_id = r.id
        WHERE COALESCE(s.ema_success, 0.5) < 0.3 AND COALESCE(s.ema_negative, 0.5) > 0.7
        """
    ).fetchall()

    for r in bad_rules:
        topic = _row_get(r, "topic_id", "general")
        base_name = _row_get(r, "name", "rule")
        name = _canonical_rule_name("retry", base_name)

        rule = {
            "name": name,
            "topic_id": topic,
            "priority": 70,  # don’t rely on r['priority'] since we didn’t select it
            "cooldown_seconds": 7200,
            "max_per_day": 2,
            "condition": {"cond": {"exists": f"signal:{topic}"}},
            "tone_strategy": {"default": "gentle"},
            "context_template": f"{topic} triggered",
            "source": "ema_failure",
            "confidence": 0.7,
            "score": 0.7,
        }
        if name in seen_names:
            continue
        rationale = (
            f"Rule '{base_name}' for topic '{topic}' shows low success "
            f"(ema_success={_row_get(r,'ema_success',0.0):.2f}) and high negatives "
            f"(ema_negative={_row_get(r,'ema_negative',0.0):.2f}). Proposing gentler retry."
        )
        _log_audit(conn, rule, rationale, {"bad_rule": dict(r)})
        seen_names.add(name)
        proposals.append(rule)

    # === Heuristic 3: evening ignore patterns → propose softer evening rule ===
    # Make timestamps robust to schema differences.
    ts_expr = "COALESCE(timestamp, fired_at, created_at)"
    if not _table_exists(conn, "rule_history"):
        logger.debug("rule_history not found; skipping time-of-day adaptation.")
    else:
        history = conn.execute(
            f"""
            SELECT topic_id,
                   outcome,
                   strftime('%H', {ts_expr}) AS hour,
                   COUNT(*) AS n
            FROM rule_history
            WHERE {ts_expr} > datetime('now','-7 day')
            GROUP BY topic_id, outcome, hour
            """
        ).fetchall()

        for h in history:
            try:
                hour = int(_row_get(h, "hour", "0"))
            except Exception:
                continue
            if _row_get(h, "outcome", "") in ("ignore", "angry") and hour >= 21:
                topic = _row_get(h, "topic_id", "general")
                name = _canonical_rule_name(f"{topic}_evening_soft", f"h{hour}")

                rule = {
                    "name": name,
                    "topic_id": topic,
                    "priority": 70,
                    "cooldown_seconds": 10800,
                    "max_per_day": 2,
                    "condition": {
                        "cond": {
                            "all": [
                                {"eq": ["signal:hour_of_day", hour]},
                                {"exists": f"signal:{topic}"},
                            ]
                        }
                    },
                    "tone_strategy": {"default": "gentle"},
                    "context_template": f"{topic} evening reminder",
                    "source": "tod_ignore_pattern",
                    "confidence": 0.65,
                    "score": 0.66,
                }
                if name in seen_names:
                    continue
                rationale = (
                    f"Last 7 days: topic '{topic}' ignored/rejected { _row_get(h,'n',0) } times around hour {hour}. "
                    "Proposing softer evening rule."
                )
                _log_audit(conn, rule, rationale, {"history": dict(h)})
                seen_names.add(name)
                proposals.append(rule)

    # Rank proposals by score (desc), then by confidence
    proposals.sort(key=lambda r: (r.get("score", 0.0), r.get("confidence", 0.0)), reverse=True)
    return proposals


def insert_proposed_rules(conn: sqlite3.Connection, rules: List[Dict[str, Any]]) -> None:
    """
    Insert proposed rules idempotently. Expects each rule to contain:
    name, topic_id, condition, tone_strategy; optional priority, cooldown_seconds, max_per_day, context_template.
    """
    for r in rules:
        conn.execute(
            """
            INSERT INTO engagement_rules
              (name, topic_id, priority, cooldown_seconds, max_per_day,
               condition_json, tone_strategy_json, context_template, enabled)
            VALUES (?,?,?,?,?,?,?,?,1)
            ON CONFLICT(name) DO NOTHING
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
            ),
        )
    conn.commit()
