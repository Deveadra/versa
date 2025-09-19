from __future__ import annotations
import math
import json
from loguru import logger
from database.sqlite import SQLiteConn

HALF_LIFE_DAYS = 30.0
LN2 = math.log(2)

class HabitMiner:
    def __init__(self, conn: SQLiteConn):
        self.conn = conn

    def _decay_factor(self, days_delta: float) -> float:
        return math.exp(-LN2 * (days_delta / HALF_LIFE_DAYS))

    def update_from_usage(self) -> int:
        c = self.conn.cursor()
        c.execute(
            """
            WITH recent AS (
              SELECT id, resolved_action, params_json, created_at
              FROM usage_log ORDER BY id DESC LIMIT 5000
            )
            SELECT id, resolved_action, params_json, strftime('%s', 'now') - strftime('%s', created_at) AS age_sec
            FROM recent
            WHERE resolved_action IS NOT NULL
            """
        )
        updates = 0
        rows = c.fetchall()
        for r in rows:
            try:
                params = json.loads(r[2] or "{}")
            except Exception:
                params = {}
            age_days = max(0.0, (r[3] or 0) / 86400.0)
            df = self._decay_factor(age_days)
            keys = []
            if svc := params.get("service"):
                keys.append(f"music.service={svc}")
            if genre := params.get("genre"):
                keys.append(f"music.genre={genre}")
            if greet := params.get("greeting_style"):
                keys.append(f"ux.greeting_style={greet}")
            if bedtime := params.get("sleep_time"):
                keys.append(f"user.sleep_time={bedtime}")
            for key in keys:
                c.execute("SELECT id, count, score FROM habits WHERE key = ?", (key,))
                row = c.fetchone()
                if row:
                    hid, cnt, score = row
                    new_cnt = cnt + 1
                    new_score = score * 0.99 + df
                    c.execute(
                        "UPDATE habits SET count = ?, score = ?, last_used = CURRENT_TIMESTAMP WHERE id = ?",
                        (new_cnt, new_score, hid),
                    )
                else:
                    c.execute(
                        "INSERT INTO habits (key, count, score, last_used) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                        (key, 1, df),
                    )
                updates += 1
        self.conn.conn.commit()
        logger.info(f"HabitMiner updated {updates} habit rows")
        return updates

    def top(self, prefix: str, n: int = 3):
        c = self.conn.cursor()
        c.execute(
            "SELECT key, count, score FROM habits WHERE key LIKE ? ORDER BY score DESC, count DESC LIMIT ?",
            (f"{prefix}%", n),
        )
        cols = ["key", "count", "score"]
        return [dict(zip(cols, r)) for r in c.fetchall()]
