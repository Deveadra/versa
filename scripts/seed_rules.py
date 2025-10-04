#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

import yaml  # pip install pyyaml

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "config" / "assistant.db"
RULES_PATH = BASE_DIR / "config" / "engagement_rules.yaml"


def load_rules(path: Path):
    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        import json

        return json.loads(path.read_text(encoding="utf-8"))


def seed_rules(conn, rules):
    cur = conn.cursor()
    for r in rules:
        cur.execute(
            """
            INSERT INTO engagement_rules
              (name, topic_id, priority, cooldown_seconds, max_per_day,
               condition_json, tone_strategy_json, context_template, enabled)
            VALUES (?,?,?,?,?,?,?,?,1)
            ON CONFLICT(name) DO UPDATE SET
              topic_id=excluded.topic_id,
              priority=excluded.priority,
              cooldown_seconds=excluded.cooldown_seconds,
              max_per_day=excluded.max_per_day,
              condition_json=excluded.condition_json,
              tone_strategy_json=excluded.tone_strategy_json,
              context_template=excluded.context_template,
              updated_at=datetime('now')
        """,
            (
                r["name"],
                r["topic_id"],
                r.get("priority", 50),
                r.get("cooldown_seconds", 1800),
                r.get("max_per_day", 6),
                json.dumps(r["condition"]),
                json.dumps(r["tone_strategy"]),
                r.get("context_template", ""),
            ),
        )
    conn.commit()


def main():
    rules = load_rules(RULES_PATH)
    if not rules:
        print("No rules found in file.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    seed_rules(conn, rules)
    print(f"Seeded {len(rules)} rules into {DB_PATH}")


if __name__ == "__main__":
    main()
