from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ScoreboardSnapshot:
    ts: str
    dream_runs_total: int
    dream_runs_passed: int
    prs_opened_total: int
    last_run_status: str

class Scoreboard:
    def __init__(self, conn) -> None:
        self.conn = conn

    def bump(self, key: str, delta: int = 1) -> None:
        self.conn.execute(
            "INSERT INTO scoreboard(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=value+?",
            (key, delta, delta),
        )
        self.conn.commit()
